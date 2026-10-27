"""A bird's story for children, told on the kiosk when its bird is tapped.

A language model writes a handful of short chunks per species and language, from
where the frame hangs, the month, and what the station has heard of the bird.
ElevenLabs reads a chunk aloud the first time someone asks it to. Both are kept
under the frame's data folder, so a species costs one generation until it is
redone from /stories.

Configured from the environment only, since keys do not belong in settings.json:

    FUGLERAMME_STORY_URL      an OpenAI-compatible API (default https://api.openai.com/v1)
    FUGLERAMME_STORY_KEY      its key; with no key or model there are no stories
    FUGLERAMME_STORY_MODEL    the model to ask
    FUGLERAMME_STORY_PLACE    where the frame hangs, in words ("near Hannover, Germany")
    FUGLERAMME_STORY_SECONDS  how long each chunk stays up (default 8)
    ELEVENLABS_API_KEY        with none, nothing is read aloud
    ELEVENLABS_VOICE_ID       the voice (default a premade multilingual one)
    ELEVENLABS_MODEL          the speech model (default eleven_multilingual_v2)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .languages import NONE, SCIENTIFIC, STATION, station_locale
from .names import canonical, normalize
from .source import Source, Unavailable

log = logging.getLogger(__name__)

FOLDER = "stories"
_TIMEOUT = 60  # seconds: a model writing a page is slow, but not this slow
_RETRY = 600  # seconds before a failed species is asked for again
_CHUNKS = (4, 12)  # what a reply has to hold to be a story
ELEVENLABS = "https://api.elevenlabs.io/v1"

_PROMPT = """You write short texts about birds for children aged 4 to 9, \
shown a chunk at a time beside a picture of the bird on a picture frame at home. \
Write in the language with the BCP 47 code "{language}".

Reply with JSON only: {{"chunks": ["...", "..."]}} holding 6 to 9 chunks. Each \
chunk is one or two short, simple sentences, at most 25 words, that read on \
their own. Speak to the child directly and warmly.

In this order, cover:
- which bird this is and how to recognise it, by look and by its song or call
- why it is here at this place and time of year: whether it lives here all year, \
is a winter guest, a summer visitor that breeds here, or passing through on \
migration, and if it migrates, where it goes and why
- what it eats and where it finds it
- one surprising fact a child will remember
- a last chunk inviting the child to listen or look out for it

Only state what you are sure of. If its status at this place is uncertain, say \
so gently. No numbers of detections, no dates, no emojis, no markdown."""

_FACTS = """Bird: {common} ({scientific}).
Where the frame hangs: {place}.
This month: {month}.
What the frame's microphone has heard: {heard}."""


@dataclass(frozen=True)
class Config:
    url: str = "https://api.openai.com/v1"
    key: str = ""
    model: str = ""
    place: str = ""
    seconds: float = 8.0
    voice_key: str = ""
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    voice_model: str = "eleven_multilingual_v2"

    @property
    def writes(self) -> bool:
        return bool(self.key and self.model)

    @property
    def speaks(self) -> bool:
        return self.writes and bool(self.voice_key)

    @property
    def speaker(self) -> str:
        """The voice and model a chunk is read in, so a new one is spoken afresh
        rather than served from the old one's recordings."""
        return quote(f"{self.voice_id}-{self.voice_model}", safe="")

    @classmethod
    def from_env(cls) -> Config:
        env = os.environ.get
        d = cls()
        try:
            seconds = max(2.0, float(env("FUGLERAMME_STORY_SECONDS", "") or d.seconds))
        except ValueError:
            seconds = d.seconds
        return cls(
            url=(env("FUGLERAMME_STORY_URL") or d.url).rstrip("/"),
            key=env("FUGLERAMME_STORY_KEY", ""),
            model=env("FUGLERAMME_STORY_MODEL", ""),
            place=env("FUGLERAMME_STORY_PLACE", ""),
            seconds=seconds,
            voice_key=env("ELEVENLABS_API_KEY", ""),
            voice_id=env("ELEVENLABS_VOICE_ID") or d.voice_id,
            voice_model=env("ELEVENLABS_MODEL") or d.voice_model,
        )


@dataclass(frozen=True)
class Story:
    species: str
    language: str
    chunks: tuple[str, ...]
    model: str
    written_at: str


def language_of(primary: str) -> str:
    """What a story is written in: the frame's first language, the station's for
    STATION, and English where the names are scientific."""
    if primary == STATION:
        return station_locale().split("-")[0] or "en"
    return "en" if primary in (SCIENTIFIC, NONE) else primary


def _post(url: str, headers: dict[str, str], body: dict) -> bytes:
    request = Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers}
    )
    try:
        with urlopen(request, timeout=_TIMEOUT) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read()[:300].decode(errors="replace")
        raise Unavailable(f"{url} answered {error.code}: {detail}") from error
    except (URLError, OSError) as error:
        raise Unavailable(f"{url} unreachable: {error}") from error


def chunks_of(reply: str) -> tuple[str, ...]:
    """The chunks out of a model's reply, which may wrap its JSON in a code fence."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    try:
        found = json.loads(text).get("chunks")
    except (ValueError, AttributeError) as error:
        raise ValueError(f"not the JSON asked for: {reply[:200]!r}") from error
    if not isinstance(found, list) or not all(isinstance(c, str) for c in found):
        raise ValueError(f"no list of chunks: {reply[:200]!r}")
    kept = tuple(c.strip() for c in found if c.strip())
    if not _CHUNKS[0] <= len(kept) <= _CHUNKS[1]:
        raise ValueError(f"{len(kept)} chunks, wanted {_CHUNKS[0]} to {_CHUNKS[1]}")
    return kept


class Stories:
    """The stories on disk, and one thread writing the missing ones."""

    def __init__(self, data_dir: Path, source: Source, config: Config | None = None):
        self.config = config or Config.from_env()
        self._dir = data_dir / FOLDER
        self._source = source
        self._lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, str, str, bool]] = queue.Queue()
        self._pending: set[tuple[str, str]] = set()
        self._failed: dict[tuple[str, str], tuple[float, str]] = {}
        self._worker: threading.Thread | None = None

    # -- disk ----------------------------------------------------------------

    def _path(self, species: str, language: str) -> Path:
        return self._dir / language / f"{normalize(species)}.json"

    def _voices(self, species: str, language: str) -> Path:
        return self._dir / language / normalize(species)

    def get(self, species: str, language: str) -> Story | None:
        try:
            raw = json.loads(self._path(species, language).read_text())
            return Story(**{**raw, "chunks": tuple(raw["chunks"])})
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _save(self, story: Story) -> None:
        path = self._path(story.species, story.language)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(asdict(story), ensure_ascii=False, indent=1))
        partial.replace(path)
        voices = self._voices(story.species, story.language)
        if recordings := len(list(voices.rglob("*.mp3"))):
            log.info(
                "Dropped %d recordings of the old %s story of %s",
                recordings,
                story.language,
                story.species,
            )
        shutil.rmtree(voices, ignore_errors=True)

    # -- state -----------------------------------------------------------------

    def state(self, species: str, language: str, name_of: Callable[[str], str] = str) -> str:
        """ "off", "ready", "pending" or "failed"; asking starts a missing story."""
        if not self.config.writes:
            return "off"
        if self.get(species, language):
            return "ready"
        self.prepare([species], language, name_of)
        with self._lock:
            if (canonical(species), language) in self._pending:
                return "pending"
        return "failed"

    def failure(self, species: str, language: str) -> str:
        with self._lock:
            return self._failed.get((canonical(species), language), (0.0, ""))[1]

    def prepare(
        self,
        species: list[str],
        language: str,
        name_of: Callable[[str], str] = str,
        redo: bool = False,
    ) -> None:
        """Queue the stories not on disk yet, or `redo` them, each with the
        bird's name in `language`. A failure is not asked again for `_RETRY`
        seconds, so a broken key costs one call per bird."""
        if not self.config.writes:
            return
        with self._lock:
            for name in map(canonical, species):
                key = (name, language)
                failed_at = self._failed.get(key, (0.0, ""))[0]
                if key in self._pending or (not redo and self.get(name, language)):
                    continue
                if not redo and time.monotonic() - failed_at < _RETRY and failed_at:
                    continue
                self._failed.pop(key, None)
                self._pending.add(key)
                self._queue.put((name, language, name_of(name), redo))
                log.info("Queued the %s story of %s%s", language, name, " (redo)" if redo else "")
            if self._worker is None and self._pending:
                self._worker = threading.Thread(target=self._work, name="stories", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        while True:
            name, language, common, redo = self._queue.get()
            started = time.monotonic()
            try:
                told = self.write(name, language, common)
                self._save(told)
                log.info(
                    "Wrote the %s story of %s with %s: %d chunks in %.1f s",
                    language,
                    name,
                    self.config.model,
                    len(told.chunks),
                    time.monotonic() - started,
                )
                if redo:
                    self._speak(told)
            except (Unavailable, ValueError) as error:
                log.warning(
                    "No %s story of %s: %s (not asked again for %d min)",
                    language,
                    name,
                    error,
                    _RETRY // 60,
                )
                with self._lock:
                    self._failed[(name, language)] = (time.monotonic(), str(error))
            finally:
                with self._lock:
                    self._pending.discard((name, language))

    def _speak(self, story: Story) -> None:
        """Read a redone story aloud at once, so whoever asked for it in the
        admin hears the new wording without waiting on the first press of play."""
        if not self.config.speaks:
            return
        try:
            for index in range(len(story.chunks)):
                self.voice(story.species, story.language, index)
        except Unavailable as error:
            log.warning(
                "Reading the %s story of %s aloud failed: %s", story.language, story.species, error
            )

    # -- writing ---------------------------------------------------------------

    def heard(self, species: str) -> str:
        """What the station knows of the bird, in plain words for the prompt."""
        try:
            first = next(
                (
                    s.first_seen
                    for s in self._source.life_list()
                    if canonical(s.scientific_name) == species
                ),
                None,
            )
            week = dict(self._source.species_since(24 * 7)).get(species, 0)
            hours = sorted(
                {
                    d.detected_at.astimezone().hour
                    for d in self._source.recent(500)
                    if d.scientific_name == species
                }
            )
        except Unavailable:
            return "nothing known right now"
        parts = [f"first heard here in {first:%B %Y}" if first else "never heard here before"]
        parts.append(
            "heard on several days this week" if week > 3 else "heard now and then this week"
        )
        if hours:
            parts.append(f"usually heard between {hours[0]}:00 and {hours[-1] + 1}:00")
        return "; ".join(parts)

    def write(self, species: str, language: str, common: str = "") -> Story:
        config = self.config
        facts = _FACTS.format(
            common=common or species,
            scientific=species,
            place=config.place or "unknown",
            month=f"{datetime.now().astimezone():%B}",
            heard=self.heard(species),
        )
        body = {
            "model": config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _PROMPT.format(language=language)},
                {"role": "user", "content": facts},
            ],
        }
        raw = _post(
            f"{config.url}/chat/completions", {"Authorization": f"Bearer {config.key}"}, body
        )
        try:
            reply = json.loads(raw)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ValueError(f"unreadable answer: {raw[:200]!r}") from error
        return Story(
            species,
            language,
            chunks_of(reply),
            config.model,
            datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    # -- reading aloud ---------------------------------------------------------

    def voice(self, species: str, language: str, index: int) -> bytes | None:
        """One chunk read aloud as MP3, spoken once and kept. None when the
        chunk is not there or nothing reads aloud."""
        story = self.get(species, language)
        if not self.config.speaks or story is None or not 0 <= index < len(story.chunks):
            return None
        config = self.config
        path = self._voices(species, language) / config.speaker / f"{index}.mp3"
        where = f"chunk {index + 1}/{len(story.chunks)} of the {language} story of {species}"
        with self._voice_lock:
            if path.exists():
                log.info("Read %s from disk (%s)", where, config.speaker)
                return path.read_bytes()
            started = time.monotonic()
            text = story.chunks[index]
            audio = _post(
                f"{ELEVENLABS}/text-to-speech/{quote(config.voice_id)}?output_format=mp3_44100_128",
                {"xi-api-key": config.voice_key, "Accept": "audio/mpeg"},
                {"text": text, "model_id": config.voice_model},
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
            log.info(
                "Spoke %s with ElevenLabs (%s): %d characters, %d kB in %.1f s",
                where,
                config.speaker,
                len(text),
                len(audio) // 1024,
                time.monotonic() - started,
            )
            return audio

    def listing(
        self, species: list[str], language: str, name_of: Callable[[str], str]
    ) -> list[dict]:
        """For /stories: each species with its story, newest heard first."""
        rows = []
        for name in species:
            story = self.get(name, language)
            with self._lock:
                pending = (canonical(name), language) in self._pending
            rows.append(
                {
                    "species": name,
                    "name": name_of(name),
                    "state": "pending" if pending else "ready" if story else "missing",
                    "chunks": list(story.chunks) if story else [],
                    "written_at": story.written_at if story else "",
                    "failure": self.failure(name, language),
                }
            )
        return rows

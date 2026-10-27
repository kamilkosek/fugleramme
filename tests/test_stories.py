"""A tapped bird's story: written once per species and language by a model,
read aloud on request, kept on disk, and redone only when asked."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from fugleramme import story
from fugleramme.settings import Settings, SettingsStore
from fugleramme.status import Status
from fugleramme.story import Config, Stories, chunks_of
from fugleramme.web import server

BLACKBIRD = "Turdus merula"
TIT = "Parus major"  # heard by the fake at seed 0, which never hears a blackbird
CHUNKS = [f"Chunk {n}." for n in range(6)]
PASSWORD = "wren-house"


@pytest.fixture
def ai():
    """A fake OpenAI-compatible model and a fake ElevenLabs, recording what they were asked."""
    asked: list[dict] = []
    spoken: list[str] = []
    answer = {"reply": json.dumps({"chunks": CHUNKS}), "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/chat/completions"):
                asked.append(body)
                payload = json.dumps({"choices": [{"message": {"content": answer["reply"]}}]})
                self._send(answer["status"], payload.encode(), "application/json")
            else:
                spoken.append(body["text"])
                self._send(200, b"ID3 " + body["text"].encode(), "audio/mpeg")

        def _send(self, status, body, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    original = story.ELEVENLABS
    story.ELEVENLABS = url + "/eleven"
    try:
        yield url, asked, spoken, answer
    finally:
        story.ELEVENLABS = original
        httpd.shutdown()


def _config(url: str, voice: bool = True) -> Config:
    return Config(
        url=url + "/v1", key="k", model="m", place="a garden", voice_key="v" if voice else ""
    )


def _until(check, seconds: float = 5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if found := check():
            return found
        time.sleep(0.02)
    raise AssertionError("never happened")


def test_a_reply_in_a_code_fence_still_reads():
    assert chunks_of('```json\n{"chunks": ["a", "b", "c", "d"]}\n```') == ("a", "b", "c", "d")


@pytest.mark.parametrize("reply", ["no json", '{"chunks": "one"}', '{"chunks": ["a", "b"]}'])
def test_a_reply_that_is_not_a_story_is_refused(reply):
    with pytest.raises(ValueError):
        chunks_of(reply)


def test_a_story_is_written_once_and_redone_only_when_asked(tmp_path, source, ai):
    url, asked, _spoken, _answer = ai
    stories = Stories(tmp_path, source(count=40, seed=0), _config(url))

    assert stories.state(BLACKBIRD, "de", lambda _: "Amsel") == "pending"
    told = _until(lambda: stories.get(BLACKBIRD, "de"))
    assert told.chunks == tuple(CHUNKS) and stories.state(BLACKBIRD, "de") == "ready"
    facts = asked[0]["messages"][1]["content"]
    assert "Amsel (Turdus merula)" in facts and "a garden" in facts
    assert '"de"' in asked[0]["messages"][0]["content"]

    stories.prepare([BLACKBIRD], "de")
    assert len(asked) == 1  # on disk: nothing to ask
    assert Stories(tmp_path, source(), _config(url)).get(BLACKBIRD, "de") == told

    stories.prepare([BLACKBIRD], "de", redo=True)
    _until(lambda: len(asked) == 2 and stories.state(BLACKBIRD, "de") == "ready")


def test_a_redo_is_read_aloud_at_once(tmp_path, source, ai):
    url, _asked, spoken, _answer = ai
    stories = Stories(tmp_path, source(count=40, seed=0), _config(url))
    stories.prepare([BLACKBIRD], "de")
    _until(lambda: stories.get(BLACKBIRD, "de"))
    assert spoken == []  # a first story waits for someone to press play

    stories.prepare([BLACKBIRD], "de", redo=True)
    _until(lambda: len(spoken) == len(CHUNKS))
    assert spoken == CHUNKS


def test_a_failed_story_says_why_and_is_not_asked_again_at_once(tmp_path, source, ai):
    url, asked, _spoken, answer = ai
    answer["status"] = 401
    stories = Stories(tmp_path, source(count=40, seed=0), _config(url))

    stories.prepare([BLACKBIRD], "de")
    _until(lambda: stories.failure(BLACKBIRD, "de"))
    assert "401" in stories.failure(BLACKBIRD, "de")
    assert stories.state(BLACKBIRD, "de") == "failed"
    assert len(asked) == 1


def test_a_chunk_is_spoken_once_and_kept(tmp_path, source, ai):
    url, _asked, spoken, _answer = ai
    stories = Stories(tmp_path, source(count=40, seed=0), _config(url))
    stories.prepare([BLACKBIRD], "de")
    _until(lambda: stories.get(BLACKBIRD, "de"))

    assert stories.voice(BLACKBIRD, "de", 2) == b"ID3 Chunk 2."
    assert stories.voice(BLACKBIRD, "de", 2) == b"ID3 Chunk 2."
    assert spoken == ["Chunk 2."]
    assert stories.voice(BLACKBIRD, "de", 99) is None

    changed = Stories(tmp_path, source(), dataclasses.replace(_config(url), voice_id="other"))
    assert changed.voice(BLACKBIRD, "de", 2) == b"ID3 Chunk 2."
    assert spoken == ["Chunk 2.", "Chunk 2."]  # a new voice is spoken afresh


def test_with_no_key_there_are_no_stories_and_nothing_reads_aloud(tmp_path, source, ai):
    url, asked, _spoken, _answer = ai
    stories = Stories(tmp_path, source(), Config(url=url))
    assert stories.state(BLACKBIRD, "de") == "off"
    assert not Config(key="k", model="m").speaks
    assert asked == []


@pytest.mark.parametrize(
    "primary,language", [("de", "de"), ("sci", "en"), ("", "en"), ("nb", "nb")]
)
def test_a_story_is_told_in_the_frame_s_first_language(primary, language):
    assert story.language_of(primary) == language


# -- over the wire -------------------------------------------------------------


@pytest.fixture
def told(tmp_path, source, ai, monkeypatch):
    """A served frame with stories switched on, and a password on its admin."""
    url, *_ = ai
    monkeypatch.setenv("FUGLERAMME_STORY_URL", url + "/v1")
    monkeypatch.setenv("FUGLERAMME_STORY_KEY", "k")
    monkeypatch.setenv("FUGLERAMME_STORY_MODEL", "m")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "v")
    style = tmp_path / "images" / "classic" / "birds"
    style.mkdir(parents=True)
    Image.new("RGBA", (120, 90), (40, 40, 40, 255)).save(style / "parus-major.png")
    store = SettingsStore(
        tmp_path / "s.json",
        Settings(admin_password=PASSWORD, require_sign_in=True, primary_language="de"),
    )
    handler = server.make_handler(
        source(count=40, seed=0), tmp_path / "images", store, None, Status()
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as error:
        return error.status, error.headers["Content-Type"], error.read()


def _story(base: str, species: str = TIT) -> dict:
    return json.loads(_get(f"{base}/story?species={urllib.parse.quote(species)}")[2])


def test_the_kiosk_gets_a_heard_bird_s_story_its_voice_and_its_picture(told):
    _until(lambda: _story(told)["state"] == "ready")
    answer = _story(told)
    assert (
        answer["chunks"] == CHUNKS
        and answer["voice"] == "21m00Tcm4TlvDq8ikWAM-eleven_multilingual_v2"
    )
    assert answer["title"] == TIT and answer["language"] == "de"  # no dictionary offline

    quoted = urllib.parse.quote(TIT)
    status, kind, body = _get(f"{told}/story/voice?species={quoted}&n=0")
    assert (status, kind, body) == (200, "audio/mpeg", b"ID3 Chunk 0.")
    status, kind, _body = _get(f"{told}/story/plate?species={quoted}")
    assert (status, kind) == (200, "image/png")


def test_a_bird_the_station_never_heard_costs_nothing(told, ai):
    _url, asked, _spoken, _answer = ai
    assert _get(f"{told}/story?species=Aptenodytes%20forsteri")[0] == 404
    assert _get(f"{told}/story/voice?species=Aptenodytes%20forsteri&n=0")[0] == 404
    assert asked == []


def test_the_stories_page_and_a_redo_are_behind_the_password(told):
    assert _get(told + "/stories.json")[0] == 401
    request = urllib.request.Request(
        told + "/stories", data=f"species={TIT}".encode(), method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=10)
    assert refused.value.status == 401


def test_a_voice_that_fails_is_a_502_and_leaves_the_story_up(told, monkeypatch):
    _until(lambda: _story(told)["state"] == "ready")
    monkeypatch.setattr(story, "ELEVENLABS", "http://127.0.0.1:9")  # nothing listens there
    quoted = urllib.parse.quote(TIT)
    assert _get(f"{told}/story/voice?species={quoted}&n=0")[0] == 502
    assert _story(told)["state"] == "ready"

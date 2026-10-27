# Bird stories

Tap a bird on the kiosk and its page opens: the bird on one side, a short text
about it for children on the other, and its best recorded call plays once. The
text comes a sentence or two at a time, with a bar below running down until the
next. On a portrait screen the bird sits above the text.

The texts are written by a language model of your choice, once per species and
language, and kept in the frame's data folder. A small play button reads them
aloud through ElevenLabs, a chunk at a time, and the bar then follows the voice.

## Setting it up

Stories are off until a model is named. Add these to the container's
environment:

```yaml
    environment:
      FUGLERAMME_STORY_URL: https://api.openai.com/v1  # any OpenAI-compatible API
      FUGLERAMME_STORY_KEY: sk-...
      FUGLERAMME_STORY_MODEL: <model name>                # as your API names it
      FUGLERAMME_STORY_PLACE: near Hannover, Germany     # where the frame hangs, in words
      FUGLERAMME_STORY_SECONDS: 8                        # each chunk, when not read aloud
      ELEVENLABS_API_KEY: ...                            # optional: reading aloud
      ELEVENLABS_VOICE_ID: 21m00Tcm4TlvDq8ikWAM          # optional: the voice
      ELEVENLABS_MODEL: eleven_multilingual_v2           # optional
```

Any API that speaks OpenAI's `/chat/completions` works: OpenAI, Mistral,
OpenRouter, or Ollama on your own network (`http://<host>:11434/v1`).

The story is written in the frame's first language (see [Languages](languages.md)),
in English where names are scientific. The model is told where the frame hangs,
the month, and what the microphone has heard of the bird, never your coordinates.

Only birds the station has heard get a story, so nobody on the network can spend
the key on other birds. Stories for the birds on the page are written as soon as
they appear on it, so the first tap need not wait.

## Changing a story

**Stories** in the admin's menu (`/stories`) lists every bird the station has
heard with its text. **Redo** asks the model again; the kiosk shows the new text
once it is written. Its spoken chunks are made again too, the next time someone
presses play.

The page is behind the admin password when one is set. The kiosk's own story
routes are open like the rest of the kiosk.

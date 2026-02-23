# linux-stt

Wayland speech-to-text CLI in Bun + TypeScript + AI SDK.

Stage 1 features:
- STT -> input
- STT -> clipboard

No keybind logic is built in. Your compositor should call the commands.

## Install

```bash
bun install
```

## Build Bundle

```bash
bun run build
```

This emits a single bundled CLI file at `dist/linux-stt.js` with a shebang banner.

## Commands

```bash
# Start recording (default sink: input)
bun run index.ts start

# Start recording to clipboard sink
bun run index.ts start --to clipboard

# Stop recording, transcribe, and emit to sink
bun run index.ts stop

# Check whether recording is active
bun run index.ts status
```

You can also use the script alias:

```bash
bun run stt -- start --to input
bun run stt -- stop
```

To expose a `linux-stt` command for compositor bindings:

```bash
bun link
```

## Config

Config file path:

`~/.config/linux-stt/config.json`

The app auto-creates this file on first run.

Example:

```json
{
  "provider": "groq",
  "model": "whisper-large-v3-turbo",
  "minDurationMs": 300,
  "transcriptionTimeoutMs": 30000,
  "audioFeedback": {
    "enabled": true,
    "volume": 0.12,
    "durationMs": 90,
    "startFrequencyHz": 940,
    "stopFrequencyHz": 680
  },
  "cleanup": {
    "enabled": true,
    "model": "openai/gpt-5-nano",
    "temperature": 0,
    "maxOutputTokens": 160,
    "timeoutMs": 10000
  },
  "recording": {
    "sampleRate": 16000,
    "channels": 1
  },
  "apiKeys": {
    "groq": "",
    "openai": ""
  }
}
```

Optional field:
- `language`: set a forced language code like `"en"`; omit for auto-detection.

Cleanup settings:
- `cleanup.enabled`: enable/disable LLM cleanup pass after STT.
- `cleanup.model`: provider/model-id format where everything after the first slash is model ID.
  - Examples: `openai/gpt-5-nano`, `groq/openai/gpt-oss-120b`
- `cleanup.temperature`: recommended `0` to keep output close to raw STT.

Audio feedback settings:
- `audioFeedback.enabled`: play beep on recording start/stop.
- `audioFeedback.volume`: `0.0` to `1.0`.
- `audioFeedback.durationMs`: beep duration.
- `audioFeedback.startFrequencyHz` / `audioFeedback.stopFrequencyHz`: start/stop beep tones.

API keys can come from either:
- `config.json` (`apiKeys.groq` / `apiKeys.openai`)
- environment variables (`GROQ_API_KEY` / `OPENAI_API_KEY`)

Provider/model defaults:
- `provider: "groq"` -> `model: "whisper-large-v3-turbo"`
- `provider: "openai"` -> `model: "gpt-4o-mini-transcribe"`

## Runtime dependencies (Wayland)

- `pw-record` (PipeWire)
- `wl-copy` for clipboard sink
- input sink fallback order: `wtype` -> `ydotool`
- audio feedback playback command (first available): `pw-play` -> `paplay` -> `aplay`

## Suggested compositor bindings

Hold-to-talk pattern:
- key press: `linux-stt start --to input`
- key release: `linux-stt stop`

Clipboard variant:
- key press: `linux-stt start --to clipboard`
- key release: `linux-stt stop`

Adjust exact binding syntax in your compositor config.

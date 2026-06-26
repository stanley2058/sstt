# sstt

Speech-to-text CLI in Bun + TypeScript + AI SDK.

Stage 1 features:
- STT -> input
- STT -> clipboard

No keybind logic is built in. Your compositor, launcher, or hotkey tool should call the commands.

## Install

```bash
bun install
```

## Build Bundle

```bash
bun run build
```

This emits a single bundled CLI file at `dist/sstt.js` with a shebang banner.

## Commands

```bash
# Start recording (default sink: input)
bun run index.ts start

# Start recording to clipboard sink
bun run index.ts start --to clipboard

# Toggle recording state (idle -> start, recording -> stop)
bun run index.ts toggle --to input

# Stop recording, transcribe, and emit to sink
bun run index.ts stop

# Check whether recording is active
bun run index.ts status

# Print config and state paths
bun run index.ts paths
```

You can also use the script alias:

```bash
bun run stt -- start --to input
bun run stt -- toggle --to clipboard
bun run stt -- stop
bun run stt -- paths
```

To expose an `sstt` command for keybindings:

```bash
bun link
```

## Config

Config file path:

- Linux: `~/.config/sstt/config.json`
- macOS: `~/Library/Application Support/sstt/config.json`

Runtime state path:

- Linux: `~/.local/state/sstt`
- macOS: `~/Library/Caches/sstt`

The app auto-creates the config file on first run.

You can print the active paths at any time with:

```bash
sstt paths
```

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
  "desktopNotifications": {
    "enabled": true,
    "reminderIntervalMs": 300000
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

Desktop notification settings:
- `desktopNotifications.enabled`: show start/stop notifications and active-recording reminders.
- `desktopNotifications.reminderIntervalMs`: reminder interval while recording; set `0` to disable reminders while keeping start/stop notifications.

API keys can come from either:
- `config.json` (`apiKeys.groq` / `apiKeys.openai`)
- environment variables (`GROQ_API_KEY` / `OPENAI_API_KEY`)

Provider/model defaults:
- `provider: "groq"` -> `model: "whisper-large-v3-turbo"`
- `provider: "openai"` -> `model: "gpt-4o-mini-transcribe"`

## Runtime dependencies

### Linux

- `pw-record` (PipeWire)
- `wl-copy` for clipboard sink
- input sink fallback order: `wtype` -> `ydotool`
- audio feedback playback command (first available): `pw-play` -> `paplay` -> `aplay`
- desktop notifications (first available): `notify-send` -> `kdialog`

### macOS

- `ffmpeg` for audio capture
- `pbcopy` / `pbpaste` for clipboard integration
- `osascript` for input sink paste
- `osascript` for desktop notifications
- `afplay` for audio feedback

macOS input sink behavior:
- `--to clipboard`: copy transcript to the clipboard
- `--to input`: copy transcript, send `Cmd+V` to the frontmost app, then restore the previous text clipboard

macOS permissions:
- microphone permission for the terminal or launcher running `sstt`
- Accessibility / Automation permission for the terminal or launcher if you use `--to input`

## Suggested keybindings

Linux hold-to-talk pattern:
- key press: `sstt start --to input`
- key release: `sstt stop`

Linux clipboard variant:
- key press: `sstt start --to clipboard`
- key release: `sstt stop`

Single-key toggle alternative:
- key press: `sstt toggle --to input`
- key press: `sstt toggle --to clipboard`

Adjust exact binding syntax in your compositor or launcher config.

On macOS, use a hotkey tool that does not steal focus from the target app before `sstt stop` runs, since `--to input` pastes into the current frontmost app.

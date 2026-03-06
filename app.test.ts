import { describe, expect, test } from "bun:test";

import {
  buildMacPasteAppleScript,
  getAppPaths,
  getRecordingCommand,
  getSupportedPlatform,
  renderHelp,
  renderPaths,
  selectPlaybackCommand,
} from "./app.ts";

const config = {
  provider: "groq" as const,
  model: "whisper-large-v3-turbo",
  minDurationMs: 300,
  transcriptionTimeoutMs: 30000,
  audioFeedback: {
    enabled: true,
    volume: 0.12,
    durationMs: 90,
    startFrequencyHz: 940,
    stopFrequencyHz: 680,
  },
  cleanup: {
    enabled: false,
    model: "openai/gpt-5-nano" as const,
    temperature: 0,
    maxOutputTokens: 160,
    timeoutMs: 10000,
  },
  recording: {
    sampleRate: 16000,
    channels: 1,
  },
};

describe("platform helpers", () => {
  test("supports linux and macOS", () => {
    expect(getSupportedPlatform("linux")).toBe("linux");
    expect(getSupportedPlatform("darwin")).toBe("darwin");
  });

  test("rejects unsupported platforms", () => {
    expect(() => getSupportedPlatform("win32")).toThrow(/Unsupported platform/);
  });

  test("builds linux paths", () => {
    const paths = getAppPaths("linux", "/tmp/home");
    expect(paths.configPath).toBe("/tmp/home/.config/sstt/config.json");
    expect(paths.stateDir).toBe("/tmp/home/.local/state/sstt");
  });

  test("builds macOS paths", () => {
    const paths = getAppPaths("darwin", "/tmp/home");
    expect(paths.configPath).toBe("/tmp/home/Library/Application Support/sstt/config.json");
    expect(paths.stateDir).toBe("/tmp/home/Library/Caches/sstt");
  });
});

describe("command builders", () => {
  test("builds linux recording command", () => {
    const command = getRecordingCommand("linux", config, "/tmp/audio.wav");
    expect(command.command).toBe("pw-record");
    expect(command.args).toEqual(["--rate", "16000", "--channels", "1", "--format", "s16", "/tmp/audio.wav"]);
  });

  test("builds macOS recording command", () => {
    const command = getRecordingCommand("darwin", config, "/tmp/audio.wav");
    expect(command.command).toBe("ffmpeg");
    expect(command.args).toContain("avfoundation");
    expect(command.args).toContain(":default");
    expect(command.args).toContain("/tmp/audio.wav");
  });

  test("selects platform playback command", () => {
    expect(selectPlaybackCommand("darwin", (name) => name === "afplay")).toEqual({
      command: "afplay",
      args: [],
    });
    expect(selectPlaybackCommand("linux", (name) => name === "paplay")).toEqual({
      command: "paplay",
      args: [],
    });
  });
});

describe("help surfaces paths", () => {
  test("renders paths output", () => {
    const text = renderPaths("darwin", getAppPaths("darwin", "/tmp/home"));
    expect(text).toContain("platform=darwin");
    expect(text).toContain("config=/tmp/home/Library/Application Support/sstt/config.json");
  });

  test("renders help with paths command", () => {
    const text = renderHelp("darwin", getAppPaths("darwin", "/tmp/home"));
    expect(text).toContain("sstt paths");
    expect(text).toContain('Run "sstt paths"');
  });
});

test("builds macOS paste script", () => {
  expect(buildMacPasteAppleScript()).toContain('keystroke "v" using command down');
});

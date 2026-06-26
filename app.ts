import { experimental_transcribe as transcribe, generateText } from "ai";
import { createGroq } from "@ai-sdk/groq";
import { createOpenAI } from "@ai-sdk/openai";
import { copyFile, mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

type Sink = "input" | "clipboard";
type ProviderName = "groq" | "openai";
type ProviderModelSpec = `${ProviderName}/${string}`;
type SupportedPlatform = "linux" | "darwin";

type AudioFeedbackConfig = {
  enabled: boolean;
  volume: number;
  durationMs: number;
  startFrequencyHz: number;
  stopFrequencyHz: number;
};

type CleanupConfig = {
  enabled: boolean;
  model: ProviderModelSpec;
  temperature: number;
  maxOutputTokens: number;
  timeoutMs: number;
};

type DesktopNotificationsConfig = {
  enabled: boolean;
  reminderIntervalMs: number;
};

type AppConfig = {
  provider: ProviderName;
  model?: string;
  language?: string;
  minDurationMs: number;
  transcriptionTimeoutMs: number;
  audioFeedback: AudioFeedbackConfig;
  desktopNotifications: DesktopNotificationsConfig;
  cleanup: CleanupConfig;
  recording: {
    sampleRate: number;
    channels: number;
  };
  apiKeys?: {
    groq?: string;
    openai?: string;
  };
};

type SessionState = {
  pid: number;
  indicatorPid?: number;
  sink: Sink;
  startedAt: string;
  audioPath: string;
  sampleRate: number;
  channels: number;
};

type AppPaths = {
  configDir: string;
  configPath: string;
  stateDir: string;
  sessionPath: string;
  recordingsDir: string;
};

type CommandSpec = {
  command: string;
  args: string[];
};

type PlaybackCommand = CommandSpec;
type BeepType = "start" | "stop";

const APP_NAME = "sstt";
const LEGACY_APP_NAME = "linux-stt";
const MACOS_PASTE_RESTORE_DELAY_MS = 175;
const DEFAULT_REMINDER_INTERVAL_MS = 5 * 60 * 1000;

const DEFAULT_CONFIG: AppConfig = {
  provider: "groq",
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
  desktopNotifications: {
    enabled: true,
    reminderIntervalMs: DEFAULT_REMINDER_INTERVAL_MS,
  },
  cleanup: {
    enabled: false,
    model: "openai/gpt-5-nano",
    temperature: 0,
    maxOutputTokens: 160,
    timeoutMs: 10000,
  },
  recording: {
    sampleRate: 16000,
    channels: 1,
  },
};

export function getSupportedPlatform(platform: NodeJS.Platform = process.platform): SupportedPlatform {
  if (platform === "linux" || platform === "darwin") {
    return platform;
  }

  throw new Error(
    `Unsupported platform "${platform}". ${APP_NAME} currently supports Linux and macOS.`,
  );
}

export function getAppPaths(
  platform: SupportedPlatform = getSupportedPlatform(),
  homePath = homedir(),
): AppPaths {
  return getAppPathsForName(APP_NAME, platform, homePath);
}

function getAppPathsForName(
  appName: string,
  platform: SupportedPlatform,
  homePath: string,
): AppPaths {
  if (platform === "darwin") {
    const configDir = join(homePath, "Library", "Application Support", appName);
    const stateDir = join(homePath, "Library", "Caches", appName);

    return {
      configDir,
      configPath: join(configDir, "config.json"),
      stateDir,
      sessionPath: join(stateDir, "session.json"),
      recordingsDir: join(stateDir, "recordings"),
    };
  }

  const configDir = join(homePath, ".config", appName);
  const stateDir = join(homePath, ".local", "state", appName);

  return {
    configDir,
    configPath: join(configDir, "config.json"),
    stateDir,
    sessionPath: join(stateDir, "session.json"),
    recordingsDir: join(stateDir, "recordings"),
  };
}

const PLATFORM = getSupportedPlatform();
const APP_PATHS = getAppPaths(PLATFORM);
const LEGACY_APP_PATHS = getAppPathsForName(LEGACY_APP_NAME, PLATFORM, homedir());

function getRecordingBackendDescription(platform: SupportedPlatform): string {
  return platform === "darwin" ? "ffmpeg (AVFoundation)" : "pw-record";
}

function getClipboardBackendDescription(platform: SupportedPlatform): string {
  return platform === "darwin" ? "pbcopy" : "wl-copy";
}

function getInputBackendDescription(platform: SupportedPlatform): string {
  return platform === "darwin"
    ? "clipboard paste via pbcopy + osascript (restores previous text clipboard)"
    : "wtype -> ydotool";
}

export function renderHelp(
  platform: SupportedPlatform = PLATFORM,
  paths: AppPaths = APP_PATHS,
): string {
  return `${APP_NAME}

Usage:
  ${APP_NAME} start [--to input|clipboard]
  ${APP_NAME} toggle [--to input|clipboard]
  ${APP_NAME} stop
  ${APP_NAME} status
  ${APP_NAME} paths
  ${APP_NAME} help

Notes:
  - Start begins recording with ${getRecordingBackendDescription(platform)}.
  - Toggle starts when idle, stops when recording.
  - Stop ends recording, transcribes audio, and sends output to your selected sink.
  - Clipboard sink uses ${getClipboardBackendDescription(platform)}.
  - Input sink uses ${getInputBackendDescription(platform)}.
  - Config lives at ${paths.configPath}
  - Runtime state lives at ${paths.stateDir}
  - Run "${APP_NAME} paths" to print all paths again.`;
}

export function renderPaths(
  platform: SupportedPlatform = PLATFORM,
  paths: AppPaths = APP_PATHS,
): string {
  return [
    `platform=${platform}`,
    `config=${paths.configPath}`,
    `state=${paths.stateDir}`,
    `session=${paths.sessionPath}`,
    `recordings=${paths.recordingsDir}`,
  ].join("\n");
}

function printHelp(): void {
  console.log(renderHelp());
}

function printPaths(): void {
  console.log(renderPaths());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asPositiveNumber(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return fallback;
  }

  return value;
}

function asNonNegativeNumber(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return fallback;
  }

  return value;
}

function asNumberInRange(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }

  if (value < min) {
    return min;
  }

  if (value > max) {
    return max;
  }

  return value;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function parseProviderModelSpec(spec: string): { provider: ProviderName; modelId: string } {
  const separatorIndex = spec.indexOf("/");
  if (separatorIndex < 1 || separatorIndex === spec.length - 1) {
    throw new Error(
      `Invalid model spec "${spec}". Expected format provider/model-id, e.g. openai/gpt-5-nano.`,
    );
  }

  const providerPart = spec.slice(0, separatorIndex);
  if (providerPart !== "groq" && providerPart !== "openai") {
    throw new Error(`Unsupported provider "${providerPart}" in model spec "${spec}"`);
  }

  return {
    provider: providerPart,
    modelId: spec.slice(separatorIndex + 1),
  };
}

function parseConfig(raw: unknown): AppConfig {
  if (!isRecord(raw)) {
    return structuredClone(DEFAULT_CONFIG);
  }

  const providerRaw = raw.provider;
  const provider: ProviderName = providerRaw === "openai" ? "openai" : "groq";

  const recordingRaw = isRecord(raw.recording) ? raw.recording : {};
  const apiKeysRaw = isRecord(raw.apiKeys) ? raw.apiKeys : {};
  const cleanupRaw = isRecord(raw.cleanup) ? raw.cleanup : {};
  const audioFeedbackRaw = isRecord(raw.audioFeedback) ? raw.audioFeedback : {};
  const desktopNotificationsRaw = isRecord(raw.desktopNotifications)
    ? raw.desktopNotifications
    : {};

  const cleanupModelRaw = asOptionalString(cleanupRaw.model) ?? DEFAULT_CONFIG.cleanup.model;
  parseProviderModelSpec(cleanupModelRaw);
  const cleanupModel = cleanupModelRaw as ProviderModelSpec;

  const parsed: AppConfig = {
    provider,
    model: asOptionalString(raw.model),
    language: asOptionalString(raw.language),
    minDurationMs: asPositiveNumber(raw.minDurationMs, DEFAULT_CONFIG.minDurationMs),
    transcriptionTimeoutMs: asPositiveNumber(
      raw.transcriptionTimeoutMs,
      DEFAULT_CONFIG.transcriptionTimeoutMs,
    ),
    audioFeedback: {
      enabled: asBoolean(audioFeedbackRaw.enabled, DEFAULT_CONFIG.audioFeedback.enabled),
      volume: asNumberInRange(audioFeedbackRaw.volume, DEFAULT_CONFIG.audioFeedback.volume, 0, 1),
      durationMs: asPositiveNumber(
        audioFeedbackRaw.durationMs,
        DEFAULT_CONFIG.audioFeedback.durationMs,
      ),
      startFrequencyHz: asPositiveNumber(
        audioFeedbackRaw.startFrequencyHz,
        DEFAULT_CONFIG.audioFeedback.startFrequencyHz,
      ),
      stopFrequencyHz: asPositiveNumber(
        audioFeedbackRaw.stopFrequencyHz,
        DEFAULT_CONFIG.audioFeedback.stopFrequencyHz,
      ),
    },
    desktopNotifications: {
      enabled: asBoolean(
        desktopNotificationsRaw.enabled,
        DEFAULT_CONFIG.desktopNotifications.enabled,
      ),
      reminderIntervalMs: asNonNegativeNumber(
        desktopNotificationsRaw.reminderIntervalMs,
        DEFAULT_CONFIG.desktopNotifications.reminderIntervalMs,
      ),
    },
    cleanup: {
      enabled: asBoolean(cleanupRaw.enabled, DEFAULT_CONFIG.cleanup.enabled),
      model: cleanupModel,
      temperature: asNonNegativeNumber(
        cleanupRaw.temperature,
        DEFAULT_CONFIG.cleanup.temperature,
      ),
      maxOutputTokens: asPositiveNumber(
        cleanupRaw.maxOutputTokens,
        DEFAULT_CONFIG.cleanup.maxOutputTokens,
      ),
      timeoutMs: asPositiveNumber(cleanupRaw.timeoutMs, DEFAULT_CONFIG.cleanup.timeoutMs),
    },
    recording: {
      sampleRate: asPositiveNumber(
        recordingRaw.sampleRate,
        DEFAULT_CONFIG.recording.sampleRate,
      ),
      channels: asPositiveNumber(recordingRaw.channels, DEFAULT_CONFIG.recording.channels),
    },
    apiKeys: {
      groq: asOptionalString(apiKeysRaw.groq),
      openai: asOptionalString(apiKeysRaw.openai),
    },
  };

  if (!parsed.model) {
    parsed.model =
      parsed.provider === "openai" ? "gpt-4o-mini-transcribe" : "whisper-large-v3-turbo";
  }

  return parsed;
}

async function ensureConfig(): Promise<AppConfig> {
  await mkdir(APP_PATHS.configDir, { recursive: true });

  try {
    const text = await readFile(APP_PATHS.configPath, "utf8");
    const json = JSON.parse(text) as unknown;
    return parseConfig(json);
  } catch (error) {
    const maybeNodeError = error as NodeJS.ErrnoException;
    if (maybeNodeError.code !== "ENOENT") {
      throw new Error(`Failed to read config at ${APP_PATHS.configPath}: ${String(error)}`);
    }

    await writeFile(APP_PATHS.configPath, `${JSON.stringify(DEFAULT_CONFIG, null, 2)}\n`, "utf8");
    console.log(`Created default config at ${APP_PATHS.configPath}`);
    return structuredClone(DEFAULT_CONFIG);
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    const maybeNodeError = error as NodeJS.ErrnoException;
    if (maybeNodeError.code === "ENOENT") {
      return false;
    }

    throw error;
  }
}

async function migrateLegacyPaths(): Promise<void> {
  if (
    !(await pathExists(APP_PATHS.configPath)) &&
    (await pathExists(LEGACY_APP_PATHS.configPath))
  ) {
    await mkdir(APP_PATHS.configDir, { recursive: true });
    await copyFile(LEGACY_APP_PATHS.configPath, APP_PATHS.configPath);
    console.log(`Migrated config from ${LEGACY_APP_PATHS.configPath} to ${APP_PATHS.configPath}`);
  }

  if (
    !(await pathExists(APP_PATHS.sessionPath)) &&
    (await pathExists(LEGACY_APP_PATHS.sessionPath))
  ) {
    await mkdir(APP_PATHS.stateDir, { recursive: true });
    await copyFile(LEGACY_APP_PATHS.sessionPath, APP_PATHS.sessionPath);
    console.log(`Migrated session from ${LEGACY_APP_PATHS.sessionPath} to ${APP_PATHS.sessionPath}`);
  }
}

async function ensureStateDirs(): Promise<void> {
  await mkdir(APP_PATHS.stateDir, { recursive: true });
  await mkdir(APP_PATHS.recordingsDir, { recursive: true });
}

async function writeSession(session: SessionState): Promise<void> {
  await ensureStateDirs();
  await writeFile(APP_PATHS.sessionPath, `${JSON.stringify(session, null, 2)}\n`, "utf8");
}

async function readSession(): Promise<SessionState | null> {
  try {
    const text = await readFile(APP_PATHS.sessionPath, "utf8");
    const parsed = JSON.parse(text) as unknown;
    if (!isRecord(parsed)) {
      return null;
    }

    const pid = parsed.pid;
    const indicatorPid = parsed.indicatorPid;
    const sink = parsed.sink;
    const startedAt = parsed.startedAt;
    const audioPath = parsed.audioPath;
    const sampleRate = parsed.sampleRate;
    const channels = parsed.channels;

    if (
      typeof pid !== "number" ||
      (sink !== "input" && sink !== "clipboard") ||
      typeof startedAt !== "string" ||
      typeof audioPath !== "string" ||
      typeof sampleRate !== "number" ||
      typeof channels !== "number"
    ) {
      return null;
    }

    if (indicatorPid !== undefined && typeof indicatorPid !== "number") {
      return null;
    }

    return {
      pid,
      indicatorPid,
      sink,
      startedAt,
      audioPath,
      sampleRate,
      channels,
    };
  } catch (error) {
    const maybeNodeError = error as NodeJS.ErrnoException;
    if (maybeNodeError.code === "ENOENT") {
      return null;
    }

    throw new Error(`Failed to read session file: ${String(error)}`);
  }
}

async function clearSession(): Promise<void> {
  await rm(APP_PATHS.sessionPath, { force: true });
}

function commandExists(command: string): boolean {
  const result = Bun.spawnSync(["which", command], {
    stdout: "ignore",
    stderr: "ignore",
  });

  return result.exitCode === 0;
}

export function selectPlaybackCommand(
  platform: SupportedPlatform,
  exists: (command: string) => boolean = commandExists,
): PlaybackCommand | null {
  if (platform === "darwin") {
    return exists("afplay") ? { command: "afplay", args: [] } : null;
  }

  if (exists("pw-play")) {
    return { command: "pw-play", args: [] };
  }

  if (exists("paplay")) {
    return { command: "paplay", args: [] };
  }

  if (exists("aplay")) {
    return { command: "aplay", args: [] };
  }

  return null;
}

function quoteAppleScriptString(value: string): string {
  return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"').replaceAll("\n", "\\n")}"`;
}

export function getDesktopNotificationCommand(
  platform: SupportedPlatform,
  title: string,
  body: string,
  exists: (command: string) => boolean = commandExists,
): CommandSpec | null {
  if (platform === "darwin") {
    if (!exists("osascript")) {
      return null;
    }

    return {
      command: "osascript",
      args: [
        "-e",
        `display notification ${quoteAppleScriptString(body)} with title ${quoteAppleScriptString(title)}`,
      ],
    };
  }

  if (exists("notify-send")) {
    return {
      command: "notify-send",
      args: ["-a", APP_NAME, title, body],
    };
  }

  if (exists("kdialog")) {
    return {
      command: "kdialog",
      args: ["--title", title, "--passivepopup", body, "10"],
    };
  }

  return null;
}

async function sendDesktopNotification(
  config: AppConfig,
  title: string,
  body: string,
  warnIfUnavailable = true,
): Promise<void> {
  if (!config.desktopNotifications.enabled) {
    return;
  }

  const notification = getDesktopNotificationCommand(PLATFORM, title, body);
  if (!notification) {
    if (warnIfUnavailable) {
      console.error(
        PLATFORM === "darwin"
          ? "Warning: desktop notifications enabled, but osascript was not found"
          : "Warning: desktop notifications enabled, but notify-send or kdialog was not found",
      );
    }

    return;
  }

  try {
    await runCommandQuiet(notification.command, notification.args);
  } catch (error) {
    if (warnIfUnavailable) {
      console.error(`Warning: failed to show desktop notification. ${String(error)}`);
    }
  }
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function createSineWaveWav(
  frequencyHz: number,
  durationMs: number,
  volume: number,
  sampleRate = 16000,
): Uint8Array {
  const sampleCount = Math.max(1, Math.round((durationMs / 1000) * sampleRate));
  const channels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataSize = sampleCount * channels * bytesPerSample;
  const wavSize = 44 + dataSize;
  const wav = new Uint8Array(wavSize);
  const view = new DataView(wav.buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  for (let i = 0; i < sampleCount; i += 1) {
    const t = i / sampleRate;
    const sample = Math.sin(2 * Math.PI * frequencyHz * t);
    const value = Math.max(-1, Math.min(1, sample * volume));
    const pcmValue = Math.round(value * 32767);
    view.setInt16(44 + i * 2, pcmValue, true);
  }

  return wav;
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function runCommandQuiet(command: string, args: string[]): Promise<void> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
  });

  const exitCode = await processResult.exited;
  if (exitCode !== 0) {
    throw new Error(`${command} exited with code ${exitCode}`);
  }
}

async function runCommand(command: string, args: string[]): Promise<void> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "pipe",
  });

  const stderr = processResult.stderr ? await new Response(processResult.stderr).text() : "";
  const exitCode = await processResult.exited;
  if (exitCode !== 0) {
    const message = stderr.trim();
    throw new Error(message.length > 0 ? message : `${command} exited with code ${exitCode}`);
  }
}

async function runCommandWithInput(command: string, args: string[], input: string): Promise<void> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "pipe",
    stdout: "ignore",
    stderr: "pipe",
  });

  if (!processResult.stdin) {
    throw new Error(`Unable to write to ${command} stdin`);
  }

  processResult.stdin.write(input);
  processResult.stdin.end();

  const stderr = processResult.stderr ? await new Response(processResult.stderr).text() : "";
  const exitCode = await processResult.exited;
  if (exitCode !== 0) {
    const message = stderr.trim();
    throw new Error(message.length > 0 ? message : `${command} exited with code ${exitCode}`);
  }
}

async function readCommandOutput(command: string, args: string[]): Promise<string> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdoutPromise = processResult.stdout
    ? new Response(processResult.stdout).text()
    : Promise.resolve("");
  const stderrPromise = processResult.stderr
    ? new Response(processResult.stderr).text()
    : Promise.resolve("");

  const [stdout, stderr, exitCode] = await Promise.all([
    stdoutPromise,
    stderrPromise,
    processResult.exited,
  ]);

  if (exitCode !== 0) {
    const message = stderr.trim();
    throw new Error(message.length > 0 ? message : `${command} exited with code ${exitCode}`);
  }

  return stdout;
}

async function playFeedbackTone(config: AppConfig, beepType: BeepType): Promise<void> {
  if (!config.audioFeedback.enabled) {
    return;
  }

  const playback = selectPlaybackCommand(PLATFORM);
  if (!playback) {
    console.error("Warning: audio feedback enabled, but no playback command found");
    return;
  }

  await ensureStateDirs();

  const frequencyHz =
    beepType === "start"
      ? config.audioFeedback.startFrequencyHz
      : config.audioFeedback.stopFrequencyHz;

  const wav = createSineWaveWav(
    frequencyHz,
    config.audioFeedback.durationMs,
    config.audioFeedback.volume,
  );

  const beepPath = join(APP_PATHS.stateDir, `beep-${beepType}.wav`);
  await writeFile(beepPath, wav);

  try {
    await runCommandQuiet(playback.command, [...playback.args, beepPath]);
  } catch (error) {
    console.error(`Warning: failed to play ${beepType} beep. ${String(error)}`);
  }
}

async function waitForProcessExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isProcessAlive(pid)) {
      return true;
    }

    await sleep(50);
  }

  return !isProcessAlive(pid);
}

function formatElapsedDuration(startedAt: string): string {
  const elapsedMs = Math.max(0, Date.now() - new Date(startedAt).getTime());
  const totalSeconds = Math.round(elapsedMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }

  return `${seconds}s`;
}

function startDesktopNotificationReminder(
  config: AppConfig,
  recordingPid: number,
): number | undefined {
  if (!config.desktopNotifications.enabled || config.desktopNotifications.reminderIntervalMs <= 0) {
    return undefined;
  }

  if (!getDesktopNotificationCommand(PLATFORM, "", "", commandExists)) {
    return undefined;
  }

  const scriptPath = process.argv[1];
  if (!scriptPath) {
    return undefined;
  }

  const indicator = Bun.spawn(
    [
      process.execPath,
      scriptPath,
      "indicator",
      String(recordingPid),
      String(config.desktopNotifications.reminderIntervalMs),
    ],
    {
      stdin: "ignore",
      stdout: "ignore",
      stderr: "ignore",
      detached: true,
    },
  );

  indicator.unref();
  return indicator.pid;
}

async function stopIndicator(indicatorPid: number | undefined): Promise<void> {
  if (!indicatorPid || !isProcessAlive(indicatorPid)) {
    return;
  }

  try {
    process.kill(indicatorPid, "SIGTERM");
  } catch {
    return;
  }

  await waitForProcessExit(indicatorPid, 1000);
}

function parseSinkArg(args: string[], commandName: "start" | "toggle"): Sink {
  let sink: Sink = "input";

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (!arg) {
      continue;
    }

    if (arg === "--to") {
      const value = args[i + 1];
      if (!value) {
        throw new Error("Missing value for --to. Use input or clipboard.");
      }

      if (value !== "input" && value !== "clipboard") {
        throw new Error(`Invalid --to value: ${value}`);
      }

      sink = value;
      i += 1;
      continue;
    }

    if (arg.startsWith("--to=")) {
      const value = arg.slice("--to=".length);
      if (value !== "input" && value !== "clipboard") {
        throw new Error(`Invalid --to value: ${value}`);
      }

      sink = value;
      continue;
    }

    throw new Error(`Unknown argument for ${commandName}: ${arg}`);
  }

  return sink;
}

function requireCommand(command: string, hint: string): void {
  if (!commandExists(command)) {
    throw new Error(`Missing ${command}. ${hint}`);
  }
}

export function getRecordingCommand(
  platform: SupportedPlatform,
  config: AppConfig,
  audioPath: string,
): CommandSpec {
  if (platform === "darwin") {
    return {
      command: "ffmpeg",
      args: [
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "avfoundation",
        "-i",
        ":default",
        "-ar",
        String(config.recording.sampleRate),
        "-ac",
        String(config.recording.channels),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-y",
        audioPath,
      ],
    };
  }

  return {
    command: "pw-record",
    args: [
      "--rate",
      String(config.recording.sampleRate),
      "--channels",
      String(config.recording.channels),
      "--format",
      "s16",
      audioPath,
    ],
  };
}

function getRecordingStartFailureHint(platform: SupportedPlatform): string {
  return platform === "darwin"
    ? "Check microphone permissions for your terminal or launcher and confirm ffmpeg can access the default AVFoundation input device."
    : "Check microphone permissions and PipeWire.";
}

function getRecordingMissingCommandHint(platform: SupportedPlatform): string {
  return platform === "darwin" ? "Install ffmpeg first." : "Install PipeWire tools first.";
}

async function readClipboardText(): Promise<string> {
  requireCommand("pbpaste", "macOS clipboard access requires pbpaste.");
  return readCommandOutput("pbpaste", []);
}

async function writeClipboardText(text: string): Promise<void> {
  requireCommand("pbcopy", "macOS clipboard access requires pbcopy.");
  await runCommandWithInput("pbcopy", [], text);
}

export function buildMacPasteAppleScript(): string {
  return 'tell application "System Events" to keystroke "v" using command down';
}

async function pasteTranscriptOnMac(text: string): Promise<void> {
  requireCommand("pbcopy", "macOS clipboard access requires pbcopy.");
  requireCommand("pbpaste", "macOS clipboard access requires pbpaste.");
  requireCommand(
    "osascript",
    "macOS input sink requires osascript to send Cmd+V to the frontmost app.",
  );

  const previousClipboardText = await readClipboardText();
  let clipboardReplaced = false;

  try {
    await writeClipboardText(text);
    clipboardReplaced = true;
    await runCommand("osascript", ["-e", buildMacPasteAppleScript()]);
    await sleep(MACOS_PASTE_RESTORE_DELAY_MS);
  } catch (error) {
    throw new Error(
      `macOS input sink failed. Grant Accessibility / Automation permissions to your terminal or hotkey launcher. ${String(error)}`,
    );
  } finally {
    if (!clipboardReplaced) {
      return;
    }

    try {
      await writeClipboardText(previousClipboardText);
    } catch (error) {
      console.error(
        `Warning: failed to restore previous clipboard text after paste. ${String(error)}`,
      );
    }
  }
}

async function routeTranscript(sink: Sink, text: string): Promise<void> {
  if (sink === "clipboard") {
    if (PLATFORM === "darwin") {
      await writeClipboardText(text);
      return;
    }

    requireCommand("wl-copy", "Clipboard sink needs wl-copy.");
    await runCommandWithInput("wl-copy", [], text);
    return;
  }

  if (PLATFORM === "darwin") {
    await pasteTranscriptOnMac(text);
    return;
  }

  if (commandExists("wtype")) {
    await runCommand("wtype", [text]);
    return;
  }

  if (commandExists("ydotool")) {
    await runCommandWithInput("ydotool", ["type", "--file", "-"], text);
    return;
  }

  throw new Error("Input sink needs wtype or ydotool");
}

function getApiKey(config: AppConfig, provider: ProviderName): string {
  if (provider === "openai") {
    const key = config.apiKeys?.openai ?? process.env.OPENAI_API_KEY;
    if (!key) {
      throw new Error("OPENAI_API_KEY is missing (env or config.apiKeys.openai)");
    }

    return key;
  }

  const key = config.apiKeys?.groq ?? process.env.GROQ_API_KEY;
  if (!key) {
    throw new Error("GROQ_API_KEY is missing (env or config.apiKeys.groq)");
  }

  return key;
}

function selectModel(config: AppConfig): {
  model:
    | ReturnType<ReturnType<typeof createGroq>["transcription"]>
    | ReturnType<ReturnType<typeof createOpenAI>["transcription"]>;
  providerOptions?: Record<string, Record<string, string>>;
} {
  if (config.provider === "openai") {
    const provider = createOpenAI({ apiKey: getApiKey(config, "openai") });
    return {
      model: provider.transcription(config.model ?? "gpt-4o-mini-transcribe"),
      providerOptions: config.language ? { openai: { language: config.language } } : undefined,
    };
  }

  const provider = createGroq({ apiKey: getApiKey(config, "groq") });
  return {
    model: provider.transcription(config.model ?? "whisper-large-v3-turbo"),
    providerOptions: config.language ? { groq: { language: config.language } } : undefined,
  };
}

async function cleanupTranscript(config: AppConfig, transcript: string): Promise<string> {
  const rawText = transcript.trim();
  if (!config.cleanup.enabled || rawText.length === 0) {
    return rawText;
  }

  try {
    const modelSpec = parseProviderModelSpec(config.cleanup.model);

    const model =
      modelSpec.provider === "openai"
        ? createOpenAI({ apiKey: getApiKey(config, "openai") }).languageModel(modelSpec.modelId)
        : createGroq({ apiKey: getApiKey(config, "groq") }).languageModel(modelSpec.modelId);

    const result = await generateText({
      model,
      temperature: config.cleanup.temperature,
      maxOutputTokens: config.cleanup.maxOutputTokens,
      abortSignal: AbortSignal.timeout(config.cleanup.timeoutMs),
      system:
        "You clean speech-to-text output. Keep wording, meaning, and tone as close as possible. Remove filler words, disfluencies, duplicate fragments, and obvious recognition artifacts. Keep intentional slang and names. Return only the cleaned sentence.",
      prompt: `Raw transcript:\n${rawText}\n\nCleaned transcript:`,
    });

    const cleanedText = result.text.trim();
    return cleanedText.length > 0 ? cleanedText : rawText;
  } catch (error) {
    console.error(`Warning: Transcript cleanup failed, using raw text. ${String(error)}`);
    return rawText;
  }
}

async function transcribeAudio(config: AppConfig, audioPath: string): Promise<string> {
  const audioBuffer = new Uint8Array(await Bun.file(audioPath).arrayBuffer());
  const { model, providerOptions } = selectModel(config);

  const result = await transcribe({
    model,
    audio: audioBuffer,
    abortSignal: AbortSignal.timeout(config.transcriptionTimeoutMs),
    providerOptions,
  });

  return result.text.trim();
}

async function estimateWavDurationMs(
  audioPath: string,
  sampleRate: number,
  channels: number,
): Promise<number> {
  const fileStats = await stat(audioPath);
  const totalBytes = fileStats.size;
  if (totalBytes <= 44) {
    return 0;
  }

  const pcmBytes = totalBytes - 44;
  const bytesPerSecond = sampleRate * channels * 2;
  if (bytesPerSecond <= 0) {
    return 0;
  }

  return Math.round((pcmBytes / bytesPerSecond) * 1000);
}

async function startCommand(args: string[], preloadedConfig?: AppConfig): Promise<void> {
  const config = preloadedConfig ?? (await ensureConfig());
  const sink = parseSinkArg(args, "start");

  const audioPath = join(APP_PATHS.recordingsDir, `${Date.now()}-${crypto.randomUUID()}.wav`);
  const recordingCommand = getRecordingCommand(PLATFORM, config, audioPath);

  requireCommand(
    recordingCommand.command,
    getRecordingMissingCommandHint(PLATFORM),
  );

  const existing = await readSession();
  if (existing) {
    if (isProcessAlive(existing.pid)) {
      throw new Error(
        `Recording already active (pid ${existing.pid}, sink ${existing.sink}). Use stop first.`,
      );
    }

    await stopIndicator(existing.indicatorPid);
    await clearSession();
  }

  await ensureStateDirs();
  await playFeedbackTone(config, "start");

  const recording = Bun.spawn([recordingCommand.command, ...recordingCommand.args], {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
    detached: true,
  });

  recording.unref();
  await sleep(150);

  if (!isProcessAlive(recording.pid)) {
    throw new Error(
      `Failed to start ${recordingCommand.command}. ${getRecordingStartFailureHint(PLATFORM)}`,
    );
  }

  const session: SessionState = {
    pid: recording.pid,
    sink,
    startedAt: new Date().toISOString(),
    audioPath,
    sampleRate: config.recording.sampleRate,
    channels: config.recording.channels,
  };

  await writeSession(session);
  await sendDesktopNotification(config, "sstt recording", `Recording to ${sink}`);

  const indicatorPid = startDesktopNotificationReminder(config, recording.pid);
  if (indicatorPid) {
    await writeSession({
      ...session,
      indicatorPid,
    });
  }

  console.log(`Recording started (pid ${recording.pid}, sink ${sink})`);
}

async function stopRecorder(pid: number): Promise<void> {
  if (!isProcessAlive(pid)) {
    return;
  }

  try {
    process.kill(pid, "SIGINT");
  } catch {
    return;
  }

  if (await waitForProcessExit(pid, 2000)) {
    return;
  }

  try {
    process.kill(pid, "SIGTERM");
  } catch {
    return;
  }

  if (await waitForProcessExit(pid, 2000)) {
    return;
  }

  process.kill(pid, "SIGKILL");
  await waitForProcessExit(pid, 1000);
}

async function stopCommand(): Promise<void> {
  const config = await ensureConfig();
  const session = await readSession();

  if (!session) {
    throw new Error("No active recording session");
  }

  await stopRecorder(session.pid);
  await stopIndicator(session.indicatorPid);
  await clearSession();
  await playFeedbackTone(config, "stop");
  await sendDesktopNotification(config, "sstt stopped", "Recording stopped; transcribing");

  const durationMs = await estimateWavDurationMs(
    session.audioPath,
    session.sampleRate,
    session.channels,
  );

  if (durationMs < config.minDurationMs) {
    await rm(session.audioPath, { force: true });
    throw new Error(
      `Recording too short (${durationMs}ms). Need at least ${config.minDurationMs}ms.`,
    );
  }

  try {
    const transcript = await transcribeAudio(config, session.audioPath);

    if (transcript.length === 0) {
      await rm(session.audioPath, { force: true });
      console.log("Transcription returned empty text");
      return;
    }

    const finalText = await cleanupTranscript(config, transcript);

    await routeTranscript(session.sink, finalText);
    console.log(finalText);
    await rm(session.audioPath, { force: true });
  } catch (error) {
    throw new Error(
      `Transcription or output failed. Audio kept at ${session.audioPath}. ${String(error)}`,
    );
  }
}

async function toggleCommand(args: string[]): Promise<void> {
  parseSinkArg(args, "toggle");

  const session = await readSession();
  if (session && isProcessAlive(session.pid)) {
    await stopCommand();
    return;
  }

  if (session) {
    await stopIndicator(session.indicatorPid);
    await clearSession();
  }

  const config = await ensureConfig();
  await startCommand(args, config);
}

async function statusCommand(): Promise<void> {
  const session = await readSession();
  if (!session) {
    console.log("idle");
    return;
  }

  if (!isProcessAlive(session.pid)) {
    await stopIndicator(session.indicatorPid);
    await clearSession();
    console.log("idle (cleaned stale session)");
    return;
  }

  console.log(
    `recording pid=${session.pid} sink=${session.sink} elapsed=${formatElapsedDuration(session.startedAt)}`,
  );
}

async function indicatorCommand(args: string[]): Promise<void> {
  if (args.length !== 2) {
    throw new Error("indicator requires recording pid and reminder interval");
  }

  const recordingPid = Number(args[0]);
  const reminderIntervalMs = Number(args[1]);
  if (
    !Number.isInteger(recordingPid) ||
    recordingPid <= 0 ||
    !Number.isFinite(reminderIntervalMs) ||
    reminderIntervalMs <= 0
  ) {
    throw new Error("indicator requires a positive pid and reminder interval");
  }

  while (true) {
    await sleep(reminderIntervalMs);

    const session = await readSession();
    if (!session || session.pid !== recordingPid || !isProcessAlive(recordingPid)) {
      return;
    }

    await sendDesktopNotification(
      DEFAULT_CONFIG,
      "sstt still recording",
      `Recording for ${formatElapsedDuration(session.startedAt)} to ${session.sink}`,
      false,
    );
  }
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  const command = argv[0];

  if (!command || command === "help" || command === "--help" || command === "-h") {
    printHelp();
    return;
  }

  if (command === "start") {
    await migrateLegacyPaths();
    await startCommand(argv.slice(1));
    return;
  }

  if (command === "toggle") {
    await migrateLegacyPaths();
    await toggleCommand(argv.slice(1));
    return;
  }

  if (command === "stop") {
    if (argv.length > 1) {
      throw new Error("stop does not take arguments");
    }

    await migrateLegacyPaths();
    await stopCommand();
    return;
  }

  if (command === "status") {
    if (argv.length > 1) {
      throw new Error("status does not take arguments");
    }

    await migrateLegacyPaths();
    await statusCommand();
    return;
  }

  if (command === "indicator") {
    await indicatorCommand(argv.slice(1));
    return;
  }

  if (command === "paths") {
    if (argv.length > 1) {
      throw new Error("paths does not take arguments");
    }

    printPaths();
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

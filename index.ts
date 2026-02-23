import { experimental_transcribe as transcribe } from "ai";
import { createGroq } from "@ai-sdk/groq";
import { createOpenAI } from "@ai-sdk/openai";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

type Sink = "input" | "clipboard";
type ProviderName = "groq" | "openai";

type AppConfig = {
  provider: ProviderName;
  model?: string;
  language?: string;
  minDurationMs: number;
  transcriptionTimeoutMs: number;
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
  sink: Sink;
  startedAt: string;
  audioPath: string;
  sampleRate: number;
  channels: number;
};

const APP_NAME = "linux-stt";
const CONFIG_DIR = join(homedir(), ".config", APP_NAME);
const CONFIG_PATH = join(CONFIG_DIR, "config.json");
const STATE_DIR = join(homedir(), ".local", "state", APP_NAME);
const SESSION_PATH = join(STATE_DIR, "session.json");
const RECORDINGS_DIR = join(STATE_DIR, "recordings");

const DEFAULT_CONFIG: AppConfig = {
  provider: "groq",
  model: "whisper-large-v3-turbo",
  minDurationMs: 300,
  transcriptionTimeoutMs: 30000,
  recording: {
    sampleRate: 16000,
    channels: 1,
  },
};

function printHelp(): void {
  console.log(`linux-stt

Usage:
  linux-stt start [--to input|clipboard]
  linux-stt stop
  linux-stt status
  linux-stt help

Notes:
  - Start begins recording with pw-record.
  - Stop ends recording, transcribes audio, and sends output to your selected sink.
  - Config lives at ${CONFIG_PATH}`);
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

function asOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function parseConfig(raw: unknown): AppConfig {
  if (!isRecord(raw)) {
    return structuredClone(DEFAULT_CONFIG);
  }

  const providerRaw = raw.provider;
  const provider: ProviderName = providerRaw === "openai" ? "openai" : "groq";

  const recordingRaw = isRecord(raw.recording) ? raw.recording : {};
  const apiKeysRaw = isRecord(raw.apiKeys) ? raw.apiKeys : {};

  const parsed: AppConfig = {
    provider,
    model: asOptionalString(raw.model),
    language: asOptionalString(raw.language),
    minDurationMs: asPositiveNumber(raw.minDurationMs, DEFAULT_CONFIG.minDurationMs),
    transcriptionTimeoutMs: asPositiveNumber(
      raw.transcriptionTimeoutMs,
      DEFAULT_CONFIG.transcriptionTimeoutMs,
    ),
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
  await mkdir(CONFIG_DIR, { recursive: true });

  try {
    const text = await readFile(CONFIG_PATH, "utf8");
    const json = JSON.parse(text) as unknown;
    return parseConfig(json);
  } catch (error) {
    const maybeNodeError = error as NodeJS.ErrnoException;
    if (maybeNodeError.code !== "ENOENT") {
      throw new Error(`Failed to read config at ${CONFIG_PATH}: ${String(error)}`);
    }

    await writeFile(CONFIG_PATH, `${JSON.stringify(DEFAULT_CONFIG, null, 2)}\n`, "utf8");
    console.log(`Created default config at ${CONFIG_PATH}`);
    return structuredClone(DEFAULT_CONFIG);
  }
}

async function ensureStateDirs(): Promise<void> {
  await mkdir(STATE_DIR, { recursive: true });
  await mkdir(RECORDINGS_DIR, { recursive: true });
}

async function writeSession(session: SessionState): Promise<void> {
  await ensureStateDirs();
  await writeFile(SESSION_PATH, `${JSON.stringify(session, null, 2)}\n`, "utf8");
}

async function readSession(): Promise<SessionState | null> {
  try {
    const text = await readFile(SESSION_PATH, "utf8");
    const parsed = JSON.parse(text) as unknown;
    if (!isRecord(parsed)) {
      return null;
    }

    const pid = parsed.pid;
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

    return {
      pid,
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
  await rm(SESSION_PATH, { force: true });
}

function commandExists(command: string): boolean {
  const result = Bun.spawnSync(["which", command], {
    stdout: "ignore",
    stderr: "ignore",
  });

  return result.exitCode === 0;
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

function parseStartSink(args: string[]): Sink {
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

    throw new Error(`Unknown argument for start: ${arg}`);
  }

  return sink;
}

async function runCommand(command: string, args: string[]): Promise<void> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "inherit",
  });

  const exitCode = await processResult.exited;
  if (exitCode !== 0) {
    throw new Error(`${command} exited with code ${exitCode}`);
  }
}

async function runCommandWithInput(
  command: string,
  args: string[],
  input: string,
): Promise<void> {
  const processResult = Bun.spawn([command, ...args], {
    stdin: "pipe",
    stdout: "ignore",
    stderr: "inherit",
  });

  if (!processResult.stdin) {
    throw new Error(`Unable to write to ${command} stdin`);
  }

  processResult.stdin.write(input);
  processResult.stdin.end();

  const exitCode = await processResult.exited;
  if (exitCode !== 0) {
    throw new Error(`${command} exited with code ${exitCode}`);
  }
}

async function routeTranscript(sink: Sink, text: string): Promise<void> {
  if (sink === "clipboard") {
    if (!commandExists("wl-copy")) {
      throw new Error("Clipboard sink needs wl-copy");
    }

    await runCommandWithInput("wl-copy", [], text);
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

function selectModel(config: AppConfig): { model: ReturnType<ReturnType<typeof createGroq>["transcription"]> | ReturnType<ReturnType<typeof createOpenAI>["transcription"]>; providerOptions?: Record<string, Record<string, string>> } {
  if (config.provider === "openai") {
    const apiKey = config.apiKeys?.openai ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY is missing (env or config.apiKeys.openai)");
    }

    const provider = createOpenAI({ apiKey });
    return {
      model: provider.transcription(config.model ?? "gpt-4o-mini-transcribe"),
      providerOptions: config.language ? { openai: { language: config.language } } : undefined,
    };
  }

  const apiKey = config.apiKeys?.groq ?? process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error("GROQ_API_KEY is missing (env or config.apiKeys.groq)");
  }

  const provider = createGroq({ apiKey });
  return {
    model: provider.transcription(config.model ?? "whisper-large-v3-turbo"),
    providerOptions: config.language ? { groq: { language: config.language } } : undefined,
  };
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

async function startCommand(args: string[]): Promise<void> {
  const config = await ensureConfig();
  const sink = parseStartSink(args);

  if (!commandExists("pw-record")) {
    throw new Error("Missing pw-record (install PipeWire tools)");
  }

  const existing = await readSession();
  if (existing) {
    if (isProcessAlive(existing.pid)) {
      throw new Error(
        `Recording already active (pid ${existing.pid}, sink ${existing.sink}). Use stop first.`,
      );
    }

    await clearSession();
  }

  await ensureStateDirs();

  const audioPath = join(RECORDINGS_DIR, `${Date.now()}-${crypto.randomUUID()}.wav`);
  const cmd = [
    "pw-record",
    "--rate",
    String(config.recording.sampleRate),
    "--channels",
    String(config.recording.channels),
    "--format",
    "s16",
    audioPath,
  ];

  const recording = Bun.spawn(cmd, {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
    detached: true,
  });

  recording.unref();
  await sleep(150);

  if (!isProcessAlive(recording.pid)) {
    throw new Error("Failed to start pw-record. Check microphone permissions and PipeWire.");
  }

  await writeSession({
    pid: recording.pid,
    sink,
    startedAt: new Date().toISOString(),
    audioPath,
    sampleRate: config.recording.sampleRate,
    channels: config.recording.channels,
  });

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
  await clearSession();

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

    await routeTranscript(session.sink, transcript);
    console.log(transcript);
    await rm(session.audioPath, { force: true });
  } catch (error) {
    throw new Error(
      `Transcription or output failed. Audio kept at ${session.audioPath}. ${String(error)}`,
    );
  }
}

async function statusCommand(): Promise<void> {
  const session = await readSession();
  if (!session) {
    console.log("idle");
    return;
  }

  if (!isProcessAlive(session.pid)) {
    await clearSession();
    console.log("idle (cleaned stale session)");
    return;
  }

  const elapsedMs = Date.now() - new Date(session.startedAt).getTime();
  const elapsedSeconds = Math.max(0, Math.round(elapsedMs / 1000));
  console.log(`recording pid=${session.pid} sink=${session.sink} elapsed=${elapsedSeconds}s`);
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const command = argv[0];

  if (!command || command === "help" || command === "--help" || command === "-h") {
    printHelp();
    return;
  }

  if (command === "start") {
    await startCommand(argv.slice(1));
    return;
  }

  if (command === "stop") {
    if (argv.length > 1) {
      throw new Error("stop does not take arguments");
    }

    await stopCommand();
    return;
  }

  if (command === "status") {
    if (argv.length > 1) {
      throw new Error("status does not take arguments");
    }

    await statusCommand();
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
});

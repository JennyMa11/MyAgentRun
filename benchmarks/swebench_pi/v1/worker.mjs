// Pi's native session, prompt and four tools; only tool I/O and eval budgets are adapted.
import {createInterface} from "node:readline";
import fs from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {Budget, clean, estimateInput} from "./policy.mjs";

const pending = new Map(); let requestId = 0; let initial;
const initialized = new Promise(resolve => { initial = resolve; });
const lines = createInterface({input: process.stdin});
lines.on("line", line => {
  const response = JSON.parse(line);
  if (initial) { const resolve = initial; initial = null; resolve(response); return; }
  const waiter = pending.get(response.id);
  if (waiter) { pending.delete(response.id); response.ok ? waiter.resolve(response.result) : waiter.reject(new Error(response.error)); }
});
const config = await initialized;
let secret = config.apiKey; delete config.apiKey;
const emit = value => process.stdout.write(JSON.stringify(clean(value, secret)) + "\n");
let start = performance.now();
const log = (name, data) => emit({type: "event", schema_version: "1.0", name,
  occurred_at: new Date().toISOString(), run_id: config.runId, step_id: budget.steps, data});
const budget = new Budget(config.budget);
let session; let activeCall = null; let rawUsage = null;
const io = (op, data = {}, signal) => {
  const id = ++requestId;
  const promise = new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
  emit({type: "io", id, op, ...data});
  const cancel = () => emit({type: "io", id: ++requestId, op: "cancel", target: id});
  signal?.addEventListener("abort", cancel, {once: true});
  if (signal?.aborted) cancel();
  return promise.finally(() => signal?.removeEventListener("abort", cancel));
};

try {
  process.env.PI_OFFLINE = "1"; process.env.PI_TELEMETRY = "0";
  const pi = await import(pathToFileURL(path.join(config.piRoot, "dist/index.js")));
  const {InMemoryCredentialStore} = await import(pathToFileURL(path.join(config.piRoot,
    "node_modules/@earendil-works/pi-ai/dist/auth/credential-store.js")));
  const runtime = await pi.ModelRuntime.create({credentials: new InMemoryCredentialStore(),
    modelsPath: null, modelsStorePath: path.join(config.stateDir, "models-store.json"),
    allowModelNetwork: false, refreshOnCreate: false});
  runtime.registerProvider("moerouter-eval", {api: "openai-completions", baseUrl: "https://moerouter.com/v1",
    authHeader: true, models: [{id: "glm-5.3-flash", name: "GLM 5.3 Flash", reasoning: false,
      input: ["text"], contextWindow: 65536, maxTokens: 16384,
      cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0},
      compat: {supportsDeveloperRole: false, supportsUsageInStreaming: true, maxTokensField: "max_tokens"}}]});
  await runtime.setRuntimeApiKey("moerouter-eval", secret);
  const settings = pi.SettingsManager.inMemory({compaction: {enabled: false}, retry: {enabled: false},
    cacheWarming: "off", transport: "sse"});
  const loader = {
    getExtensions: () => ({extensions: [], errors: [], runtime: pi.createExtensionRuntime()}),
    getSkills: () => ({skills: [], diagnostics: []}), getPrompts: () => ({prompts: [], diagnostics: []}),
    getThemes: () => ({themes: [], diagnostics: []}), getAgentsFiles: () => ({agentsFiles: []}),
    getSystemPrompt: () => undefined, getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => ["Resolve the supplied issue in /testbed. Network is disabled; dependencies are installed. Do not commit, alter Git metadata, or modify tests to hide a defect."],
    getAppendSystemPromptSources: () => [], extendResources: () => {}, reload: async () => {},
  };
  const tempPrefix = path.resolve(config.tempDir) + path.sep;
  const isArtifact = p => path.resolve(p).startsWith(tempPrefix);
  const checkArtifact = async p => {
    const real = await fs.realpath(p);
    if (!real.startsWith(tempPrefix) || !(await fs.stat(real)).isFile() || (await fs.stat(real)).size > 8388608)
      throw new Error("artifact_path_denied");
    return real;
  };
  const readFile = async p => isArtifact(p) ? fs.readFile(await checkArtifact(p)) :
    Buffer.from((await io("read", {path: p})).data, "base64");
  const access = async p => { if (isArtifact(p)) await checkArtifact(p); else await io("access", {path: p}); };
  const writeFile = async (p, content) => { await io("write", {path: p, content}); };
  const definitions = [
    pi.createReadToolDefinition("/testbed", {autoResizeImages: false, operations: {readFile, access}}),
    pi.createEditToolDefinition("/testbed", {operations: {readFile, writeFile, access}}),
    pi.createWriteToolDefinition("/testbed", {operations: {writeFile, mkdir: async p => { await io("mkdir", {path: p}); }}}),
    pi.createBashToolDefinition("/testbed", {exposeSessionEnvironment: false, operations: {
      exec: async (command, cwd, options) => {
        const response = await io("exec", {command, cwd, timeout: options.timeout ?? 120}, options.signal);
        options.onData(Buffer.from(response.output, "base64"));
        return {exitCode: response.exit_code};
      },
    }}),
  ];
  const tools = definitions.map(definition => ({...definition, execute: async (id, args, signal, update, ctx) => {
    budget.beforeTool();
    if (args.path) {
      const resolved = path.posix.resolve("/testbed", args.path);
      const workspace = resolved === "/testbed" || resolved.startsWith("/testbed/");
      const allowed = (workspace || (definition.name === "read" && isArtifact(args.path))) && !String(args.path).startsWith("~");
      log("permission_decided", {id, tool: definition.name, allowed, rule: "container_workspace_or_owned_output"});
      if (!allowed) throw new Error("path_outside_workspace");
    } else {
      log("permission_decided", {id, tool: definition.name, allowed: true, rule: "container_only_bash"});
    }
    log("tool_call_started", {id, tool: definition.name, arguments: args});
    const began = performance.now();
    try {
      const result = await definition.execute(id, args, signal, update, ctx);
      log("tool_call_completed", {id, tool: definition.name, status: result.isError ? "failed" : "succeeded", duration_ms: performance.now() - began});
      return result;
    } catch (error) {
      log("tool_call_completed", {id, tool: definition.name, status: "failed", error: String(error), duration_ms: performance.now() - began});
      throw error;
    }
  }}));
  const model = runtime.getModel("moerouter-eval", "glm-5.3-flash");
  ({session} = await pi.createAgentSession({cwd: "/testbed", agentDir: config.stateDir, modelRuntime: runtime,
    model, thinkingLevel: "off", settingsManager: settings, resourceLoader: loader,
    sessionManager: pi.SessionManager.inMemory("/testbed"), tools: ["read", "bash", "edit", "write"], customTools: tools}));
  session.agent.toolExecution = "sequential";
  const originalStream = session.agent.streamFunction.bind(session.agent);
  session.agent.streamFunction = async (selected, context, options) => {
    const estimated = estimateInput(context);
    log("context_built", {strategy: "full_history", estimated_tokens: estimated, limit: 65536});
    if (estimated > 65536) { budget.reason = "context_input_budget_exhausted"; throw new Error(budget.reason); }
    budget.beforeModel(); rawUsage = null;
    activeCall = {step: budget.steps, started: performance.now()};
    log("model_call_started", {model: selected.id, temperature: 0, max_output_tokens: 16384});
    return originalStream(selected, context, {...options, temperature: 0, maxTokens: 16384, maxRetries: 0,
      timeoutMs: 300000, onProviderStreamEvent: async (chunk, m) => {
        if (chunk?.usage) rawUsage = chunk.usage;
        else if (chunk?.choices?.[0]?.usage) rawUsage = chunk.choices[0].usage;
        await options?.onProviderStreamEvent?.(chunk, m);
      }});
  };
  session.subscribe(event => {
    if (event.type === "message_end" && event.message.role === "assistant" && activeCall) {
      const usage = budget.settle(rawUsage);
      log("model_call_completed", {duration_ms: performance.now() - activeCall.started,
        usage_known: usage !== null, ...usage, stop_reason: event.message.stopReason,
        error: event.message.errorMessage ?? null, pi_usage: event.message.usage});
      activeCall = null;
      if (budget.reason) session.agent.abort();
    }
    if (event.type !== "message_update" && event.type !== "tool_execution_update") log("pi_event", event);
  });
  start = performance.now();
  log("run_started", {pi_version: pi.VERSION, system_prompt: session.systemPrompt,
    tools: definitions.map(({name, description, parameters}) => ({name, description, parameters})),
    context: "full_history", compaction: false, retry: false});
  const timer = setTimeout(() => { budget.reason ||= "wallclock_budget_exhausted"; session.agent.abort(); },
    Math.max(0, config.maxSeconds * 1000 - (performance.now() - start)));
  try { await session.prompt(config.task, {expandPromptTemplates: false}); }
  finally { clearTimeout(timer); }
  if (activeCall) { budget.settle(null); activeCall = null; }
  const last = session.messages.filter(m => m.role === "assistant").at(-1);
  const status = budget.reason?.endsWith("budget_exhausted") ? "budget_exhausted" :
    budget.unknown || last?.stopReason === "error" || last?.stopReason === "aborted" ? "failed" : "finished";
  const result = {status, failure_reason: budget.reason ?? last?.errorMessage ?? null, tokens: budget.tokens,
    input_tokens: budget.input, output_tokens: budget.output, cached_input_tokens: budget.cachedInput,
    unknown_usage_calls: budget.unknown, steps: budget.steps, tool_calls: budget.tools,
    duration_ms: performance.now() - start, token_overrun: Math.max(0, budget.tokens - budget.maxTokens)};
  log("run_finished", result); emit({type: "result", ...result});
} catch (error) {
  if (activeCall) budget.settle(null);
  const failed = {status: budget.reason?.endsWith("budget_exhausted") ? "budget_exhausted" : "failed",
    failure_reason: budget.reason ?? String(error), tokens: budget.tokens,
    input_tokens: budget.input, output_tokens: budget.output, cached_input_tokens: budget.cachedInput,
    unknown_usage_calls: budget.unknown, steps: budget.steps, tool_calls: budget.tools,
    duration_ms: performance.now() - start};
  log("run_finished", failed); emit({type: "result", ...failed});
} finally {
  session?.dispose(); secret = ""; lines.close(); process.stdin.destroy();
  await new Promise(resolve => process.stdout.write("", resolve));
  process.exit(0);
}

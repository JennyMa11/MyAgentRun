// Shared offline-testable controls. All monetary fields are deliberately excluded.
export class Budget {
  constructor({maxTokens, maxSteps = 200, maxTools = 400}) {
    this.maxTokens = maxTokens; this.maxSteps = maxSteps; this.maxTools = maxTools;
    this.tokens = 0; this.steps = 0; this.tools = 0; this.unknown = 0;
    this.input = 0; this.output = 0; this.cachedInput = 0;
    this.reason = null;
  }
  beforeModel() {
    if (this.reason) throw new Error(this.reason);
    if (this.tokens >= this.maxTokens || this.steps >= this.maxSteps) {
      this.reason = this.tokens >= this.maxTokens ? "token_budget_exhausted" : "step_budget_exhausted";
      throw new Error(this.reason);
    }
    this.steps += 1;
  }
  beforeTool() {
    if (this.reason || this.tokens >= this.maxTokens || this.tools >= this.maxTools) {
      this.reason ||= this.tokens >= this.maxTokens ? "token_budget_exhausted" : "tool_budget_exhausted";
      throw new Error(this.reason);
    }
    this.tools += 1;
  }
  settle(raw) {
    if (!raw || !Number.isInteger(raw.prompt_tokens) || raw.prompt_tokens <= 0 ||
        !Number.isInteger(raw.completion_tokens) || raw.completion_tokens < 0) {
      this.unknown += 1; this.reason ||= "unknown_usage";
      return null;
    }
    const input = raw.prompt_tokens, output = raw.completion_tokens;
    const cached = raw.prompt_tokens_details?.cached_tokens ?? 0;
    if (!Number.isInteger(cached) || cached < 0 || cached > input) throw new Error("invalid_cached_usage");
    this.input += input; this.output += output; this.cachedInput += cached;
    this.tokens += input + output;
    if (this.tokens >= this.maxTokens) this.reason ||= "token_budget_exhausted";
    return {input_tokens: input, output_tokens: output, cached_input_tokens: cached, total_tokens: input + output};
  }
}

export function clean(value, key) {
  const serialized = JSON.stringify(value, (name, item) => name === "cost" ? undefined : item);
  return JSON.parse(key ? serialized.split(key).join("[REDACTED]") : serialized);
}

export function estimateInput(context) {
  const length = value => [...String(value ?? "")].length;
  let characters = length(context.systemPrompt);
  for (const message of context.messages ?? []) {
    if (typeof message.content === "string") characters += length(message.content);
    else for (const block of message.content ?? []) {
      characters += length(block.text ?? block.thinking);
      if (block.type === "toolCall") characters += length(block.name) + length(JSON.stringify(block.arguments));
    }
    if (!message.content && message.sections) characters += Object.values(message.sections).reduce((n, text) => n + length(text), 0);
  }
  return Math.max(1, Math.ceil(characters / 4));
}

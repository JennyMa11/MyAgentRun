import assert from "node:assert/strict";
import test from "node:test";
import {Budget, clean, estimateInput} from "./policy.mjs";

test("cached input is a subset, failed-call consumption and in-flight excess remain charged", () => {
  const b = new Budget({maxTokens: 100});
  b.beforeModel(); b.settle({prompt_tokens: 80, completion_tokens: 30, prompt_tokens_details: {cached_tokens: 60}});
  assert.equal(b.tokens, 110); assert.equal(b.cachedInput, 60);
  assert.throws(() => b.beforeModel(), /token_budget_exhausted/);
  assert.throws(() => b.beforeTool(), /token_budget_exhausted/);
  assert.equal(b.steps, 1); assert.equal(b.tools, 0);
});
test("missing usage and error placeholder zeros do not become zero consumption", () => {
  for (const raw of [null, {prompt_tokens: 0, completion_tokens: 0}]) {
    const b = new Budget({maxTokens: 100}); b.beforeModel();
    assert.equal(b.settle(raw), null); assert.equal(b.unknown, 1);
    assert.throws(() => b.beforeModel(), /unknown_usage/);
  }
});
test("steps and tools are stopped before additional execution", () => {
  const b = new Budget({maxTokens: 100, maxSteps: 1, maxTools: 1});
  b.beforeModel(); b.settle({prompt_tokens: 1, completion_tokens: 1});
  b.beforeTool(); assert.throws(() => b.beforeTool(), /tool_budget_exhausted/);
  const c = new Budget({maxTokens: 100, maxSteps: 1}); c.beforeModel();
  assert.throws(() => c.beforeModel(), /step_budget_exhausted/);
});
test("artifacts redact configured secrets and omit monetary accounting", () => {
  assert.deepEqual(clean({text: "private-key", usage: {cost: {total: 5}, totalTokens: 3}}, "private-key"),
    {text: "[REDACTED]", usage: {totalTokens: 3}});
});
test("input admission uses Unicode characters over four including native tool arguments", () => {
  assert.equal(estimateInput({messages:[{role:"user",content:"😀😀😀😀"}]}), 1);
  assert.equal(estimateInput({messages:[{role:"system",content:"",sections:{main:"abcdefgh"}}]}), 2);
  assert.ok(estimateInput({messages:[{content:[{type:"toolCall",name:"read",arguments:{path:"x"}}]}]}) > 1);
});

# MoeRouter Context/SSE 冒烟 v3

输入仍为 v1/v2 的三个调试任务，任务预算仍是 1,000,000 Token、200 步、400 调用、
3,600 秒。新增显式输入上下文上限 65,536，单响应输出上限 16,384，SSE 传输，300 秒超时。
这些变更来自实际发现的默认 8K 限制和连接断开，见 ADR 0033（swebench-context-and-native-environments，作者本地记录）。

前轮 `benchmark-results/swebench-smoke/v2/moerouter-high-budget/` 的原始 Trace
保留 1 次未知用量事件。路由商逐调用日志补算已确认该次消耗 10,744 Token，
前轮实际总消耗 132,192。`usage-reconciliation.json` 另存来源 hash、请求 ID 和匹配方法，
不改写原始事实。该轮结果为 Flask 补丁通过、pytest 未通过、SymPy 未运行，不隐藏失败。

本轮整轮剩余额度固定为 `3,000,000 - 132,192 = 2,867,808`，避免将调试消耗清零。
若再遇未知用量则继续停止并审计。不得覆写旧目录或丢弃失败消耗。

本轮输出目录为 `benchmark-results/swebench-smoke/v3/moerouter-context-sse/`，
完成情况以该目录的真实报告为准。后续 50 题不包含这些调试题，且须逐题完成环境门禁。

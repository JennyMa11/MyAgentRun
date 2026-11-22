# Pi 1.5.0 与 MyAgentRun 的 Lite20 历史对照

用户明确选择只运行 Pi 20 题，对照已完成的 MyAgentRun 11/20，不重跑历史基线。
范围与隔离决定见 ADR 0036（pi-historical-comparison，作者本地记录）。

固定输入沿用 `benchmarks/swebench_lite/v1/selection.json` 的全部20题，不按结果换题。
逐任务初始提交、原生 ARM64 镜像、1 CPU / 1 GB / 128 PID、断网与无挂载、
900 秒独立验证和官方判据完全复用，20题参考通过/空补丁失败的实际证据 hash 重新核对。

Pi 使用已安装并冻结的 `@earendil-works/pi-coding-agent` 1.5.0，复制完整依赖与 Node
可执行文件至受控 artifact；不修改用户全局 Pi 设置和凭据。使用 SDK 原生 AgentSession、
原生四工具及默认系统提示，工具 operations 仅改为容器执行。
同模型 `glm-5.3-flash`、同 endpoint、temperature 0、SSE、单次输出上限16384。
每题100万Token、整轮2000万Token、200次模型步骤、400次工具调用、3600秒。
Full History，模型上下文声明65536；无 Skills、MCP、外部扩展、自动压缩、重试或预热。
另按Unicode文本与工具参数字符数/4估算输入，超过65536时在发出请求前停止；
与本项目使用同阈值和估算单位，原生消息/工具表示不同，因此不宣称逐Token完全等价。
这不是 Pi 全部默认功能的验收，也不是单一机制消融。

Token 以原始 provider usage 输入+输出核算，缓存输入是输入子集，不再累加；失败消耗与
在途超量保留。未知用量单独计数，完整实际消耗返回 null，并停止后续题目。

运行中修订：第11题出现无法核对用量的连接错误后，按
ADR0036 的续跑修订（作者本地记录）允许跨批次继续未尝试题目。
每个含未知用量的历史任务另预留100万Token用于额度调度，实际消费和整轮预算满足保持未知。
原始冻结协议与Trace保留，最终报告必须披露此项偏离；不重跑失败题。
工具原始输出最多4MiB，单次桥接请求最多80000字节，超限返回明确错误；
Pi原生截断策略继续生效，其受控大输出可由read只读访问。
工作区文件 realpath 校验防止越界与符号链接；子命令通过进程组清理，最终移除整个容器。
模型调用预算从发出前检查，达到Token预算后不再调用模型或执行工具。

正式题目每题一次尝试；无论Loop自报、失败或预算终止，均捕获补丁并在全新容器独立评分。
未运行、失败、环境故障都保留在20题分母内。未知用量不得填0，参考解与测试标签不进入Agent。
正式运行前的简单工具/连通性探针在单独 probe 目录留档，消耗单列，不混入20题。

```sh
uv run python benchmarks/swebench_pi/v1/runner.py prepare
node --test benchmarks/swebench_pi/v1/policy.test.mjs
uv run pytest tests/test_pi_benchmark.py
uv run python benchmarks/swebench_pi/v1/runner.py probe
uv run python benchmarks/swebench_pi/v1/runner.py run
```

本机产物为 `benchmark-results/pi-comparison/v1/`，包括固定协议/脚本/依赖hash、
原生Pi事件与provider用量、Patch、独立评分、sidecar、实际隔离记录及受控输出。
报告比较完整分母成功率、逐题胜负、Token下界/未知数、工具调用、模型步骤和延迟。
服务状态跨轮不同，原生提示和工具集不同，差值不能作为纯 Harness 或单机制收益的因果证据。

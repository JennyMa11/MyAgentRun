# 其余六种 Context：固定 Lite20 顺序评测 v1

2026-11-17 用户明确要求其余Context策略都运行原SWE-bench Lite子集。
采用原 `benchmarks/swebench_lite/v1/selection.json` 全部20题，不按成绩换题。
六种策略每题一次，共120次；不额外重跑Full History或Pi。

| 配置名 | 实现 | 预声明参数 |
| --- | --- | --- |
| budget | BudgetContext | input_budget=65536 |
| sliding | SlidingWindowContext | window_size=20，保留关键消息及完整工具轮次 |
| summary | SummaryCompactionContext | recent_messages=12，摘要最多4000字符 |
| retrieval | RetrievalContext(FullHistoryContext) | top5；每步重新准备受控快照 |
| retrieval_summary | RetrievalContext(SummaryCompactionContext) | recent12，摘要4000字符，top5 |
| symbol | Python符号检索＋SummaryCompactionContext | recent12，摘要4000字符，top5 |

采用现有策略实现及默认参数，不新增模型生成摘要、向量库、Skills或Memory。
SnapshotRetriever每步列出最多80条文件路径，最多读取24个文件、每个4000字符，
随后按现有排序选出top5；这些读取均经Tool Runtime和权限系统，计入工具与墙钟预算。
模型可见工具仍只有shell/git。所有策略使用同一个容器文件backend；未触发检索的策略不执行快照读取。

## 固定条件与预算

- 模型MoeRouter glm-5.3-flash；temperature0；不另发thinking开关；SSE；输出上限16384。
- 单题100万Token、200步、400次工具调用、3600秒；输入估算阈值65536。
- 每策略名义2000万Token；全实验名义1.2亿Token，失败与在途超量计入消耗，不计算货币费用。
- 原生ARM64 Docker，逐任务1CPU/1GB/128PID，断网、无挂载、cap_drop=ALL和禁止提权。
- 原始20题输入、初始提交、镜像、参考通过/空补丁失败门禁、官方评分判据及900秒独立评分均复用。
- 同题各策略相邻运行；策略顺序按原任务索引轮转，每个任务依次执行六种策略，始终只有一个任务容器。
- 固定同一核心源码hash，与历史Full History一致；本轮不修reasoning_content输入估算遗漏，防止同时变更第二项。
- 每题模型用量未知时，原Loop仍终止；下一题继续运行，并为该未知任务额外预留一题名义额度。
  未知用量不填0，完整消耗与整轮预算满足保持未知。预留不是实际消耗，也不是已证明的上界。
- 工具/模型错误、环境错误和预算终止不换题、不自动重试；评分缺失保持未知。
  基础设施失败不阻断其他未尝试任务；源码、协议或门禁变化则停止，不能混入不同版本。

## 隔离与兼容

会话和Trace保存在宿主私有artifact，Context只能通过受控backend读取当前容器的 `/testbed`。
复用现有file_worker，入口根从/workspace固定到/testbed；对Python3.8增加延迟注解与
等价的Path.is_relative_to兼容函数。backend只允许list_files/read_file，禁止写入。
不挂载宿主文件，不读取评分源和标签，不把Key传入容器、Prompt或Trace。
每次执行前验证冻结协议、源码、任务树与门禁证据；API凭据只在可信宿主模型adapter中使用。

## 运行与报告

```sh
uv run python benchmarks/swebench_context/v1/run.py prepare \
  --evaluator /path/to/official-evaluator/bin/python

# 从冻结源码启动；实际启动参数与进程记录保存在artifact中。
PYTHONPATH=/absolute/artifacts/reproduction/src \
uv run python /absolute/artifacts/reproduction/benchmarks/swebench_context/v1/run.py run \
  --root /absolute/artifacts --credentials-file /absolute/private/credentials.toml
```

准备阶段不调用模型，使用真实任务容器与离线ScriptedModel验证：读取确实来自容器、路径/敏感文件拒绝、
快照调用进入Trace并计入预算、模型工具列表没有变化。
正式执行每个策略/任务组合只有一个专属artifact目录和attempt记录；整轮run_started为独占创建。

`benchmark-results/swebench-context/v1/report.json` 和 `progress.md` 持续更新完整120行与六组摘要。
原始Trace、Patch、独立评分、sidecar和容器审计逐题保存。报告成功率、已评分覆盖率、
已知/未知Token、步骤、工具调用、Loop延迟、失败及未评分，不能把finished当通过。

先检查策略是否实际触发：压缩/丢弃条数、检索读取与retrieved_chunks均由Trace核查；未触发的机制不认定有效。
只有一次执行；Full History是历史运行，且存在服务状态与输入估算的已知限制。
结果支持这些固定任务和配置的描述性比较；没有重复和不确定性分析前，不宣称通用机制收益。
具体决策见 [ADR0037](../../../docs/adr/0037-context-lite20-sweep.md)。

## 执行中发现的评分入口错误

首批运行已执行独立测试并保存原始/归一化日志，但新入口对虚拟环境解释器调用resolve()，
跟随bin/python符号链接后丢失pyvenv.cfg上下文，导致官方评分helper导入swebench失败。
这不是模型修复失败。新入口改为保留虚拟环境解释器路径，并增加回归测试。
已冻结的运行代码、协议、Trace、Patch和原始测试结果不改写；原队列继续保存原始结果。

`repair_scores.py`使用原固定官方helper及正确虚拟环境，仅解析已完成日志，
模型调用0次、测试重跑0次。校验完成标记和原始/归一化日志hash，结果绑定run_id、
任务digest、Patch、helper及评分器hash。派生报告另存
`benchmark-results/swebench-context/v1/scoring-repair/v2/`，不能称为重新实测。
原报告的每个已读取版本按hash快照保存，评分脚本及汇总脚本也保存可复现源码。
固定脚本的watch进程为新完成任务自动补算，成绩以该派生报告和sidecar为准。

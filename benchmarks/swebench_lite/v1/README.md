# 固定 SWE-bench Lite 20 题：完整链路验证 v1

本轮已完成 20/20 唯一尝试及独立评分，修复成功 11/20（55%）。
完整报告保存在作者本地笔记中，不随仓库发布；本节保留结论与口径。
已知消耗 8,105,553 Token，另有 1 次故障调用用量未知；协议续跑偏离与未测项均在报告中保留。

2026-11-16 用户将当前目标改为“先选 SWE-bench Lite 中固定的 20 个任务，跑通完整评测链路”。
此前 Verified 50 题计划停止推进，已有调试轮与失败证据保留，不混入本轮 20 题成绩。

## 预声明范围与完成条件

本轮只验证四个纯 Python 仓库上的完整工程链路，不估计全量 Lite 成绩。
完成要求是：固定 20 个输入；20 个参考/空补丁门禁通过；每题一次真实 Agent 尝试；
每份产物独立评分；保存 Trace、Patch、判据与 run_id 绑定、完整分母及消耗报告。
任务失败计入结果；Loop `finished` 不等于修复成功，缺少执行或独立证据的题目不能冒充完成。

固定来源为官方 task repo 提交 `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d`，
其中 Lite test 记录数核对为 300。仅根据仓库和 instance_id 选择：

| 仓库 | 固定配额 |
| --- | --- |
| django/django | 8 |
| pallets/flask | 2 |
| pytest-dev/pytest | 4 |
| sympy/sympy | 6 |

按 `SHA256(seed + ':' + instance_id)` 在每个仓库内排序取前若干条，
seed 为 `myagentrun_lite20_pipeline_v1_20261116`。排除已用于三题冒烟的 ID。
不使用模型成绩、参考解或测试结果筛选；环境失败、模型失败或预算终止后不换题。
本子集强调纯 Python 环境的链路覆盖，不覆盖科学计算编译、HTTPBin 网络服务等其他环境，
不能描述为 Lite 的无偏随机抽样或完整 Harness 能力验收。

全部 ID、初始提交与完整原始任务树 hash 见 `selection.json`，选题方式见 `sampling.json`。
`prepare.py` 在独立官方 SWE-bench 工具环境运行，生成该清单与受控评分源目录；
JSON 语义应可重建，正式运行实际读取文件的 SHA-256 另存于源码清单。

## 冻结执行配置

- MoeRouter `glm-5.3-flash`，Chat Completions 兼容 adapter，temperature 0，不额外发送厂商 thinking 开关。
- shell/git 两工具、Full History、单 Agent 顺序执行；Skills、Verification、Memory、自动恢复和重试关闭。
- 单题 1,000,000 Token、200 步、400 次工具调用、3,600 秒；整轮 20,000,000 Token。
- 输入上下文上限 65,536；SSE 传输，单次输出最多 16,384 Token，流式帧累计最多 32 MiB；
  网络连接、读写超时为 300 秒；持续收到流式帧时，单次调用总时长可超过此值，
  整题仍受 3,600 秒墙钟预算约束。必须报告用量，无用量时停止，不将其解释为零消耗。
- 原生 ARM64 Docker，每个 Agent/评分容器 1 CPU、1 GB、128 进程，断网、无挂载、
  capabilities 全移除、禁止提权。实际 inspect 记录到每次执行目录。
- 每题独立评分超时 900 秒，与 Agent 耗时分开统计；原始输出最多 128 MiB。
  凭据只保存在宿主私有存储，不进入镜像、工作区、Prompt 或 Trace。

依赖按 `environments.py` 中声明的可移植 profile 准备，逐题保存 Python minor、
基础镜像 digest、依赖列表、pip freeze、构建配方 hash 与最终不可变 image ID。
部分旧版 Django 使用其支持的 Python 3.8，现代版本使用 3.9/3.11；
这不是官方 AMD64 conda 镜像，因此结果是自定义环境的 20 题开发子集成绩。

## 环境与评分门禁

每道题必须由实际执行证明参考修复通过、空补丁失败，再进入模型阶段。
所有失败构建、依赖缺失、评分异常和重测尝试都保留在 `environment-attempts/`。
门禁失败不能通过删题、改阈值或给 Agent 暴露答案解决。

官方评分器固定为 SWE-bench 5.0.2 源码提交
`02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`。
完整官方测试补丁、F2P/P2P 标签与评分逻辑不变；运行原始测试命令，
只移除 conda 激活，将安装改为离线安装，补上退出码、随机完成标记，
并省略会在浅克隆中输出整个仓库的 `git show` 诊断。原始/派生脚本均保存。

原始日志以字节保留，另生成 UTF-8 解析视图。unittest 多行说明的真实完成状态
同时按规范测试名表示，保留说明文本的原标签；缺失完成状态时不推断通过。
这修复了“实际测试全部 OK，但日志解析器漏掉方法名”的问题，不放宽测试判据。
补算评分另存于带 normalizer hash 的目录，记录原始输出 hash；不得冒充重新执行测试。
归一化契约测试覆盖成功、失败、错误、跳过和未完成情况。

## 操作入口与证据

```sh
# 独立官方工具环境：固定输入，生成 sources（不进入 Agent 镜像）
python benchmarks/swebench_lite/v1/prepare.py \
  --task-repo /path/to/pinned-swe-bench-tasks \
  --output /path/to/fresh-selection \
  --artifacts /path/to/fresh-artifacts

# 项目环境：生成镜像配方，顺序构建并验证
uv run python benchmarks/swebench_lite/v1/environments.py generate
uv run python benchmarks/swebench_lite/v1/environments.py build \
  --evaluator /path/to/official-venv/bin/python

# 若只修复日志解析，可从已有原始输出再生成独立派生评分
uv run python benchmarks/swebench_lite/v1/environments.py regrade \
  --evaluator /path/to/official-venv/bin/python

# 所有门禁成功且源码、模型、协议和镜像指纹匹配后，才读取凭据并执行模型
uv run python -m myagentrun.benchmark.swebench_smoke run \
  --selection benchmarks/swebench_lite/v1/selection.json \
  --sources benchmark-results/swebench-lite/v1/sources \
  --output benchmark-results/swebench-lite/v1/baseline \
  --evaluator /path/to/official-venv/bin/python
```

本机证据根目录为 `benchmark-results/swebench-lite/v1/`，不提交到 Git。
`baseline/report.json` 保留完整 20 题结果，`predictions.jsonl` 保存捕获补丁。
`runs/<id>/` 保存原始 Trace、独立评分、sidecar、原始/归一化测试输出及资源检查。
`environment-status.json`、`gates.json`、`environment-attempts/` 保存环境门禁全过程。
`reproduction/`、`source-manifest.json`、`image-metadata.json` 固定实际未提交源码与镜像，
不能用当前分支 HEAD 代替这份快照。模型阶段禁止覆写或自动重跑已尝试的题目。

统计报告分别列出独立成功率、正常终态、预算约束、Token、工具调用与 API/模型失败。
失败消耗全部保留，未知用量停止整轮；若需从路由商日志补算，使用 `reconcile_usage.py`，
原始 Trace 和原报告不改写，补算来源、匹配规则和 hash 单独存档。
未完成的剩余题目继续保留在分母内，不依据模型成绩决定是否补跑或删除。

路由商用量补算只接受类型 2 的消耗记录，要求正输入 Token、非负输出 Token。
错误日志中的零字段不代表实际零消耗，保持未知并停止后续调用。
本轮中途状态与证据缺口记录在作者本地的进度笔记中，不随仓库发布。

2026-11-17 用户确认服务端故障并指示继续运行。按
ADR 0035（lite20-provider-failure-continuation，作者本地记录），仅豁免原第 14 题
第 30 步的未知用量停止条件，在 `continuation-01/` 继续未尝试的六题。
原清单、前 14 题尝试和评分保持不变；旧未知用量不计为零，完整 Token 消耗及预算约束保持未知。
后续新故障仍执行停止规则。最终汇总使用 `report.py --additional-root` 合并唯一尝试，
明确披露协议偏离，不把六题续跑视为前 14 题的重测。

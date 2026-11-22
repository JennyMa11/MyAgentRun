# MyAgentRun

MyAgentRun 是面向本地代码仓库的 **Coding Agent Harness / Runtime**。你可以在终端里交给 Agent 编码任务，由它读取代码、修改文件、运行测试，再通过 diff、会话记录和 Trace 查看执行过程。

- 版本：`0.3.2` · Python `>=3.12` · [Apache-2.0](LICENSE)

## 快速开始

在仓库根目录使用 [uv](https://docs.astral.sh/uv/) 安装依赖并运行离线示例。

```bash
uv sync --locked
uv run python examples/minimal_demo.py
```

这个示例使用脚本模型，在临时工作区完整执行读取、修改和测试，无需 API Key。

要在自己的仓库中打开交互式 Agent，指定工作区和项目的测试命令。

```bash
uv run myagentrun work --workspace /path/to/repo --test-command "python -m pytest"
```

`work` 默认使用 DeepSeek，可用 `--model-profile` 切换模型。密钥可以通过 `DEEPSEEK_API_KEY` 提供，也可以在首次提交任务时按提示输入。进入会话后用 `/help` 查看命令，`/diff` 查看修改，`/test` 运行配置好的测试。[模型与凭据配置](docs/model-configuration.md)

## 当前能力

- **执行与权限** · 循环跟踪步骤、Token、工具调用和墙钟预算。工具在执行前校验参数和权限，支持逐次审批和 Docker 隔离。[Runtime 说明](docs/mini-dsh-runtime.md) · [隔离与权限](docs/sandbox-security.md)
- **会话与上下文** · 支持恢复、取消和分叉，可配置历史窗口、摘要和仓库检索。原始会话记录用于恢复，选定的上下文策略决定模型每步看到的内容。[Context 策略](docs/context-engineering.md)
- **扩展与验证** · 模型、工具和运行服务通过显式接口组装。可配置 Skills、完成前验证和可选的 Project Memory。[扩展接口](docs/extensibility.md) · [Verification](docs/extension-verification.md) · [Project Memory](docs/extension-memory.md)
- **Trace 与诊断** · 记录模型、工具、权限和 Context 事件。非交互 `run` 自动输出诊断，定位超时、重复失败、用量缺口和最慢调用。独立验证结果可通过同一运行的 sidecar 加入报告。[诊断口径](docs/harness-evaluation.md#自动-trace-诊断)

## 查看一次运行

```bash
uv run myagentrun trace analyze /path/to/run.jsonl
uv run myagentrun trace replay /path/to/run.jsonl
```

`analyze` 给出问题摘要和证据位置，`replay` 按步骤还原执行过程。要核对任务是否通过独立验证，可给 `analyze` 加上 `--evidence /path/to/evidence.json`。

## 开发

源码位于 [`src/myagentrun/`](src/myagentrun/)，示例位于 [`examples/`](examples/)。架构与开发约束见 [ADR](docs/adr/README.md) 和 [AGENTS.md](AGENTS.md)，评测工具的使用方式见 [Benchmark 指南](docs/benchmark.md)。

```bash
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

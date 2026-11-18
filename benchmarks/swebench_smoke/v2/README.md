# SWE-bench 高预算冒烟协议 v2

来源：从 `../v1/selection.json` 复制同一组三个任务的完整身份和 hash，
按 2026-11-16 用户要求提高预算。没有增删任务，没有修改测试或成功判据。
JSON 的 `schema_version` 仍为数据契约 1.0；目录 v2 表示本次实验协议版本。

| 资源 | v1 单题 | v2 单题 |
| --- | --- | --- |
| 累计 Token | 50,000 | 1,000,000 |
| Loop 步数 | 48 | 200 |
| 工具调用 | 96 | 400 |
| 墙钟时间 | 600 秒 | 3,600 秒 |

三题整轮名义 Token 额度为 3,000,000。仍然顺序执行，调用中产生的 Token 超量全部记账，
达到任务或整轮额度后停止后续调用；未知用量停止整轮。不存在无限运行。
最后一题额度仍取单题额度与整轮剩余额度的最小值。

当前模型由项目 `myagentrun.toml` 选择 MoeRouter `glm-5.3-flash`。
由于相对历史 v1 同时改变模型和预算，本协议只能用于新配置的整体可用性评测，
不能将分数变化归因于单个机制。容器资源、参考/空补丁门禁、独立评分保持原要求。

生成文件可用以下代码重建（在项目根目录，输出文件须不存在）：

```python
import json
import tomllib
from pathlib import Path

settings = tomllib.loads(Path("myagentrun.toml").read_text())["myagentrun"]
selection = json.loads(Path("benchmarks/swebench_smoke/v1/selection.json").read_text())
selection.update(
    task_tokens=settings["max_tokens"],
    total_tokens=settings["max_tokens"] * len(selection["tasks"]),
    max_steps=settings["max_steps"],
    max_tool_calls=settings["max_tool_calls"],
    max_seconds=settings["max_seconds"],
)
with Path("benchmarks/swebench_smoke/v2/selection.json").open("x") as stream:
    stream.write(json.dumps(selection, indent=2) + "\n")
```

执行时传入 `--selection benchmarks/swebench_smoke/v2/selection.json`，使用新输出目录，
从已保存的不可变镜像 ID 准备 `images.json`，重新完成 `check` 后才运行 `run`。
不要修改 v1 清单、生成器或覆盖历史结果；本次仅调整额度，尚未启动 v2 模型运行。

# SWE-bench Verified 50 题整体可用性评测 v1

2026-11-16 用户要求选取 50 或 100 题，不跑全量。本轮固定 50 题；当前只完成选题，
未构建这些任务环境、未运行环境门禁、未调用模型，不存在该题集的成功率。
该清单替代原计划中尚未执行的 20 题开发集，不覆盖既有 3 题冒烟记录。

## 来源与抽样

固定官方 task repo 提交 `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d`，
元数据确认为 Verified test 的 500 条记录。排除已经用于调试的 3 个任务后，候选为 497 条。
排除清单、随机种子和仓库配额保存在 `sampling.json`，全部输入身份和任务树 hash
保存在 `selection.json`。只根据仓库和任务 ID 选择，不使用参考解、测试结果或模型成绩筛选。

每个可选仓库至少一题，其余名额按“剩余容量”比例分配，用整数最大余数法分配尾数，
余数相同时按仓库名字典序；仓库内按固定 seed 与 instance_id 的 SHA-256 排序取前若干条。
这是一份强调仓库覆盖的开发子集，不能当作官方均匀随机抽样或直接代表全量成绩。

| 仓库 | 题数 |
| --- | --- |
| astropy/astropy | 3 |
| django/django | 19 |
| matplotlib/matplotlib | 4 |
| mwaskom/seaborn | 1 |
| psf/requests | 2 |
| pydata/xarray | 3 |
| pylint-dev/pylint | 2 |
| pytest-dev/pytest | 2 |
| scikit-learn/scikit-learn | 3 |
| sphinx-doc/sphinx | 4 |
| sympy/sympy | 7 |

Verified 共 12 个仓库；Flask 唯一一题已经用于调试，因此本轮实际覆盖 11 个仓库。
环境失败或模型失败不得换题；未运行、取消或超预算题保留在完整 50 题分母中。
如将来扩大到 100 题，另行冻结扩展清单和协议，不按这 50 题的成绩选择补充题。

## 预算与执行顺序

模型沿用项目配置的 MoeRouter `glm-5.3-flash`。每题最多 1,000,000 累计 Token、
200 步、400 次工具调用、3,600 秒；50 题整轮名义上限 50,000,000 Token。
这是上限，不是实际消耗预测。逐调用记账，未知用量停止整轮；在途超量仍完整计入。

1. 先用已有三道调试题，在新模型和高预算下完成门禁与冒烟，不计入正式 50 题成绩。
2. 为这 50 题准备各自仓库与依赖；沿用原生 Docker、1 CPU/1 GB、单 Agent 顺序执行。
   现有冒烟 Dockerfile 只验证过 Flask/pytest/SymPy，不能直接用于全部仓库。
   需要适配依赖和测试激活方式，任何放宽资源或环境差异都须先记录，不能默默降级到宿主机。
3. 每题参考补丁通过、空补丁失败，固定镜像身份、评分器、源码和配置后才运行模型。
   本清单固定了输入与预算；环境尚待逐题门禁，不能描述为完整运行环境已经验收。
4. 以官方独立评分统计完整 50 题结果，报告环境故障、任务失败、预算、Token 与证据覆盖率。
   不运行机制消融，也不把不同模型与预算的结果变化解释为机制收益。

## 再生成

在独立官方 SWE-bench Python 环境运行（输出目录中的两个 JSON 必须不存在）：

```sh
python benchmarks/swebench_verified/v1/select_tasks.py \
  --task-repo /path/to/pinned-swe-bench-tasks \
  --output /path/to/fresh-selection-output
```

生成结果必须与本目录 JSON 逐字节一致。运行前用现有 `SmokeSelection` 契约验证输入，
且验证每个原始任务树 hash；独立判据与参考解不进入 Agent 的工作区或 Context。

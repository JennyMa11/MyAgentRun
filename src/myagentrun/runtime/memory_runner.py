"""Bracket the entire injected runner, including all start/end hooks and cleanup."""

import uuid

from myagentrun.domain import Message, RunResult
from myagentrun.ports import AgentRunner
from myagentrun.runtime.memory import MemoryService


class MemoryAgentRunner:
    def __init__(self, inner: AgentRunner, memory: MemoryService) -> None:
        self._inner = inner
        self._memory = memory

    async def run(
        self, task: str, *, run_id: str | None = None, prior_history: tuple[Message, ...] = ()
    ) -> RunResult:
        actual_id = run_id if run_id is not None else uuid.uuid4().hex
        self._memory.begin_run(actual_id, task=task)
        try:
            return await self._inner.run(task, run_id=actual_id, prior_history=prior_history)
        finally:
            self._memory.end_run(actual_id)

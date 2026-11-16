"""Recording adapter contracts, with deterministic sampling and no network."""

from dataclasses import dataclass, field

import pytest
from minidsh_rl.contracts import Turn
from minidsh_rl.model import RecordingModel, Sample, parse_output

from minidsh.domain import Message, ModelRequest, Role, TokenUsage


@dataclass
class Backend:
    text: str
    calls: int = 0

    def render(self, _request: ModelRequest) -> tuple[int, ...]:
        return (10, 11)

    async def sample(self, prompt: tuple[int, ...], _limit: int, _seed: int) -> Sample:
        self.calls += 1
        return Sample(Turn(prompt, (12, 13), (-0.1, -0.2)), self.text, "stop")


@dataclass
class Recorder:
    records: list[dict[str, object]] = field(default_factory=list)
    fail: bool = False

    async def append(self, record: dict[str, object]) -> None:
        if self.fail:
            raise OSError("disk failed")
        self.records.append(record)


async def test_invalid_action_is_recorded_and_returned_without_hidden_resampling() -> None:
    backend = Backend("<tool_call>broken</tool_call>")
    recorder = Recorder()
    model = RecordingModel(backend, recorder, "policy_0", "run_0")
    result = await model.complete(ModelRequest((Message(Role.USER, "fix"),)))
    assert backend.calls == 1
    assert result.recoverable_error is not None
    assert result.usage == TokenUsage(input_tokens=2, output_tokens=2)
    assert recorder.records[0]["completion_token_ids"] == [12, 13]
    assert recorder.records[0]["policy_id"] == "policy_0"


async def test_recording_failure_prevents_action_execution() -> None:
    backend = Backend('<tool_call>{"name":"shell","arguments":{"argv":["pwd"]}}</tool_call>')
    model = RecordingModel(backend, Recorder(fail=True), "policy_0", "run_0")
    with pytest.raises(OSError, match="disk failed"):
        await model.complete(ModelRequest((Message(Role.USER, "fix"),)))
    assert backend.calls == 1


def test_tool_json_is_not_reencoded_as_generated_tokens() -> None:
    text = '<tool_call>{ "name": "shell", "arguments": {"argv": ["python3", "-V"]}}</tool_call>'
    response = parse_output(text, "call_1")
    assert response.tool_calls[0].arguments == {"argv": ["python3", "-V"]}
    assert response.tool_calls[0].call_id == "call_1"


def test_terminal_end_of_turn_marker_is_framing_not_output() -> None:
    """The sampler ends every turn with `<|im_end|>`; it must not break the envelope.

    Regression: the first real positive-control v2 rollout produced three
    well-formed `<tool_call>` blocks, all rejected as `trailing text after tool
    envelope`, so `tool_calls` stayed 0 and the run failed with `invalid_action`.
    The marker is added by the chat template, not chosen by the model.
    """
    body = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert parse_output(body + "<|im_end|>", "call").tool_calls[0].name == "x"
    assert parse_output(body + "<|im_end|>\n", "call").tool_calls[0].name == "x"
    # A checkpoint can emit it twice; both are framing, so both go.
    assert parse_output(body + "<|im_end|><|im_end|>", "call").tool_calls[0].name == "x"


def test_end_of_turn_marker_inside_the_envelope_is_still_rejected() -> None:
    """Only terminal markers are framing; an interior one is model error."""
    with pytest.raises(ValueError, match="tool"):
        parse_output('<tool_call>{"name":"x",<|im_end|>"arguments":{}}</tool_call>', "call")
    with pytest.raises(ValueError, match="tool"):
        parse_output(
            '<tool_call>{"name":"x","arguments":{}}</tool_call> then <|im_end|> more', "call"
        )


def test_end_of_turn_marker_is_preserved_in_plain_prose() -> None:
    """A model that writes the marker as text keeps that text: nothing is invented."""
    text = "I cannot finish.<|im_end|>"
    response = parse_output(text, "call")
    assert response.tool_calls == ()
    assert response.content == "I cannot finish."


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"x","arguments":{}}</tool_call> trailing',
        '<tool_call>{"name":"x","arguments":{}}</tool_call>' * 2,
        '<tool_call>{"name":"x","arguments":{},"extra":1}</tool_call>',
    ],
)
def test_ambiguous_or_noncanonical_tool_protocol_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="tool"):
        parse_output(text, "call")


def test_qwen35_native_xml_function_protocol() -> None:
    text = (
        "<tool_call>\n<function=shell>\n<parameter=argv>\n"
        '["cat", "implementation.py"]\n</parameter>\n'
        "<parameter=cwd>\n.\n</parameter>\n</function>\n</tool_call>"
    )
    response = parse_output(text, "native")
    assert response.tool_calls[0].name == "shell"
    assert response.tool_calls[0].arguments == {"argv": ["cat", "implementation.py"], "cwd": "."}

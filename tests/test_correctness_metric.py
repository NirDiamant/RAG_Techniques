"""Offline GEval regression using real DeepEval/OpenAI serialization and scoring.

Run from the repository root in an isolated Python 3.12 environment:
    python -m pip install -r tests/requirements-evaluation.txt
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest tests/test_correctness_metric.py

The existing conftest imports unrelated tutorial dependencies. Disable it and
plugin autoload so DeepEval is first imported after the offline guards below.
"""

import ast
import asyncio
import json
import math
import socket
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def offline_metric(monkeypatch, tmp_path):
    network_attempts = []

    def deny_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Network access is forbidden in this offline test")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    monkeypatch.setenv("DEEPEVAL_DISABLE_DOTENV", "1")
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    monkeypatch.setenv("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    from deepeval.metrics import GEval
    from deepeval.models import OpenAIModel
    from deepeval.telemetry.client import NoopBackend, get_backend
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    # Keep the no-op backend at process exit, after monkeypatch restores env vars.
    assert isinstance(get_backend(), NoopBackend)

    source = Path(__file__).resolve().parents[1] / "evaluation/evalute_rag.py"
    # Execute the real declaration without importing unrelated RAG integrations.
    declarations = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "correctness_metric"
            for target in node.targets
        )
    ]
    assert len(declarations) == 1
    namespace = {"GEval": GEval, "LLMTestCaseParams": LLMTestCaseParams}
    exec(
        compile(ast.Module(body=declarations, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    metric = namespace["correctness_metric"]
    metric.model.model.close()
    requests = []

    def respond(request):
        assert request.method == "POST"
        assert str(request.url) == "https://offline.invalid/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer offline-test-key"
        body = json.loads(request.content)
        requests.append(body)
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"score": 8, "reason": "Synthetic judgment"}),
                "refusal": None,
            },
        }
        if body.get("logprobs"):
            choice["logprobs"] = {
                "content": [
                    {
                        "token": "8",
                        "logprob": math.log(0.75),
                        "bytes": [56],
                        "top_logprobs": [
                            {"token": "8", "logprob": math.log(0.75), "bytes": [56]},
                            {"token": "4", "logprob": math.log(0.25), "bytes": [52]},
                        ],
                    }
                ]
            }
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async_client = httpx.AsyncClient(transport=transport)

        def prepare(use_sol=False):
            if use_sol:
                # Control: the previous judge falls back to ordinary schema scoring.
                metric.model = OpenAIModel(model="gpt-5.6-sol")
                metric.model.model.close()
                metric.evaluation_model = metric.model.get_model_name()
            # Configure transport only; leave the real capability registry and
            # generation/scoring methods intact, including GEval's top-20 default.
            metric.model.kwargs["http_client"] = sync_client
            metric.model.async_http_client = async_client
            metric.model.base_url = "https://offline.invalid/v1"
            case = LLMTestCase(
                input="Question", expected_output="Answer", actual_output="Answer"
            )
            return metric, case, requests

        try:
            yield prepare
        finally:
            asyncio.run(async_client.aclose())
            assert not network_attempts


def measure(metric, case, mode):
    with asyncio.Runner() as runner:
        runner.get_loop()
        if mode == "a_measure":
            return runner.run(metric.a_measure(case, _show_indicator=False))
        if mode == "sync":
            metric.async_mode = False
        else:
            assert metric.async_mode is True  # Exercise the unmodified default.
        return metric.measure(case, _show_indicator=False)


@pytest.mark.parametrize("mode", ["sync", "default_async", "a_measure"])
def test_correctness_preserves_probability_weighting(offline_metric, mode):
    metric, case, requests = offline_metric()
    # (8 * .75 + 4 * .25) / 10 = .7, distinct from the unweighted 8 / 10.
    assert measure(metric, case, mode) == pytest.approx(0.7)
    assert metric.score == pytest.approx(0.7)
    assert metric.reason == "Synthetic judgment"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gpt-5.4"
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 5
    assert "temperature" not in request
    assert request.get("reasoning_effort", "none") == "none"
    assert request.get("reasoning", {}).get("effort", "none") == "none"


@pytest.mark.parametrize("mode", ["sync", "default_async", "a_measure"])
def test_sol_control_falls_back_to_unweighted_schema_score(offline_metric, mode):
    metric, case, requests = offline_metric(use_sol=True)
    assert measure(metric, case, mode) == pytest.approx(0.8)
    assert metric.reason == "Synthetic judgment"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gpt-5.6-sol"
    assert request["response_format"]["type"] == "json_schema"
    assert "logprobs" not in request
    assert "top_logprobs" not in request
    assert "temperature" not in request

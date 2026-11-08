"""LoRA routing, grounding, cache isolation and bounded repair (no model download)."""
import json

import pytest

from app.agents.llm_agent import LLMAgent
from app.core.config import Settings
from app.services import llm_agent_lora as local
from app.services.llm_conversation import build_conversation_plan


@pytest.fixture(autouse=True)
def clear_cache():
    LLMAgent.clear_cache()
    yield
    LLMAgent.clear_cache()


def facts():
    return {"question": "Comment porter mon haut ?", "items": [
        {"ref": "p1", "category": "haut", "colors": ["blanc"]}], "wardrobe_only": True}


def answer(ref="p1"):
    return json.dumps({"response_kind": "advice", "answer": "Pars de ton haut blanc.",
        "ideas": [], "item_refs": [ref], "follow_up": ""})


def test_lora_conversation_runs_with_hosted_provider_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(local, "adapter_identity", lambda path: (path, 1, 100))
    monkeypatch.setattr(local, "generate_conversation", lambda *args: calls.append(args) or answer())
    agent = LLMAgent(settings=Settings(llm_provider="none", llm_chat_backend="lora"))
    result = agent.run(task="assistant", context=facts(), instruction="style", cache_scope="user1")
    assert result["explanation"] == answer()
    assert len(calls) == 1 and agent.last_failure is None


def test_lora_does_not_enable_other_llm_tasks(monkeypatch):
    monkeypatch.setattr(local, "generate_conversation", lambda *args: pytest.fail("Local model called for another task"))
    agent = LLMAgent(settings=Settings(llm_provider="none", llm_chat_backend="lora"))
    assert agent.run(task="style_profile", context={}, instruction="style") == {"explanation": None}
    assert agent.parse_occasion("travail", {}) is None


def test_cache_is_scoped_by_account_and_adapter_version(monkeypatch):
    calls = []
    version = [1]
    monkeypatch.setattr(local, "adapter_identity", lambda path: (path, version[0], 100))
    monkeypatch.setattr(local, "generate_conversation", lambda *args: calls.append(args) or answer())
    agent = LLMAgent(settings=Settings(llm_provider="none", llm_chat_backend="lora"))
    def run(scope):
        return agent.run(task="assistant", context=facts(), instruction="style", cache_scope=scope)
    run("u1"); run("u1")
    assert len(calls) == 1
    run("u2")
    assert len(calls) == 2
    version[0] = 2
    run("u1")
    assert len(calls) == 3


def test_missing_adapter_fails_without_remote_call(tmp_path, monkeypatch):
    monkeypatch.setattr("httpx.post", lambda *a, **kw: pytest.fail("Unexpected remote fallback"))
    agent = LLMAgent(settings=Settings(llm_provider="none", llm_chat_backend="lora",
        llm_lora_adapter_dir=str(tmp_path / "missing")))
    assert agent.run(task="assistant", context=facts(), instruction="style")["explanation"] is None
    assert agent.last_failure == "local_model_not_ready"


def test_unknown_wardrobe_reference_is_repaired_once(monkeypatch):
    messages_seen = []
    def generate(messages, settings, schema=None):
        assert schema["properties"]["item_refs"]["items"]["enum"] == ["p1"]
        messages_seen.append(messages)
        return answer("invented") if len(messages_seen) == 1 else answer()
    monkeypatch.setattr(local, "_generate", generate)
    result = local.generate_conversation(build_conversation_plan(facts()), json.dumps({"facts": facts()}), Settings())
    assert json.loads(result)["item_refs"] == ["p1"]
    assert len(messages_seen) == 2
    assert messages_seen[1][-1]["role"] == "user"
    assert messages_seen[1][-1]["content"].endswith(facts()["question"])


def test_repair_instructions_are_not_displayed_as_style_advice(monkeypatch):
    outputs = [json.dumps({"answer": "Utilise uniquement les références autorisées.",
        "ideas": [], "item_refs": [], "follow_up": ""}), answer()]
    calls = []
    def generate(messages, settings, schema=None):
        calls.append(messages)
        return outputs.pop(0)
    monkeypatch.setattr(local, "_generate", generate)
    result = local.generate_conversation(build_conversation_plan(facts()), json.dumps({"facts": facts()}), Settings())
    assert json.loads(result)["answer"] == "Pars de ton haut blanc."
    assert len(calls) == 2


def test_persistent_invalid_reference_is_never_displayed(monkeypatch):
    calls = []
    monkeypatch.setattr(local, "_generate", lambda *a: calls.append(a) or answer("invented"))
    with pytest.raises(ValueError):
        local.generate_conversation(build_conversation_plan(facts()), json.dumps({"facts": facts()}), Settings())
    assert len(calls) == 2

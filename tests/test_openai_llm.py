import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from finregrag.core.llm.factory import build_llm_provider
from finregrag.core.llm.openai_llm import OpenAIChatLLM
from finregrag.core.vectorstore.base import RetrievedChunk

CHUNKS = [
    RetrievedChunk(text="The nighttime limit is R$ 1,000.00.", source="pix.md", chunk_index=0, score=0.92),
    RetrievedChunk(text="Records must be kept for 5 years.", source="aml.md", chunk_index=3, score=0.41),
]


@pytest.fixture
def invoked(monkeypatch) -> dict:
    """Replaces the OpenAI call: records the messages sent and returns the configured reply."""
    state: dict = {"reply": AIMessage(content="R$ 1,000.00 (pix.md)")}

    def fake_invoke(self, messages, *args, **kwargs):
        state["messages"] = messages
        return state["reply"]

    monkeypatch.setattr(ChatOpenAI, "invoke", fake_invoke)
    return state


def test_sends_the_system_prompt_and_the_context_with_its_sources(invoked: dict) -> None:
    llm = OpenAIChatLLM(model="gpt-4o-mini", api_key="sk-test")

    answer = llm.generate_answer("What is the nighttime limit?", CHUNKS)

    assert answer == "R$ 1,000.00 (pix.md)"
    system, user = invoked["messages"]
    assert system["role"] == "system"
    assert "ONLY based on the provided context" in system["content"]
    assert user["role"] == "user"
    assert "[Source: pix.md]\nThe nighttime limit is R$ 1,000.00." in user["content"]
    assert "[Source: aml.md]\nRecords must be kept for 5 years." in user["content"]
    assert user["content"].endswith("Question: What is the nighttime limit?")


def test_returns_plain_text_when_the_model_replies_with_content_blocks(invoked: dict) -> None:
    # langchain-core 1.x models may return a list of content blocks; the API needs a string.
    invoked["reply"] = AIMessage(
        content=[{"type": "text", "text": "The limit is R$ 1,000.00. "}, {"type": "text", "text": "Source: pix.md"}]
    )
    llm = OpenAIChatLLM(model="gpt-4o-mini", api_key="sk-test")

    answer = llm.generate_answer("What is the nighttime limit?", CHUNKS)

    assert answer == "The limit is R$ 1,000.00. Source: pix.md"


def test_factory_builds_the_openai_provider_and_rejects_unknown_ones() -> None:
    assert isinstance(build_llm_provider("openai", model="gpt-4o-mini", api_key="sk-test"), OpenAIChatLLM)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm_provider("not-a-provider")

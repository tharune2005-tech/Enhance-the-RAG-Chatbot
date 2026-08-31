"""History-aware RAG: rewrite follow-ups, retrieve, then answer with full chat history."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import extract_text

REWRITE_SYSTEM = """You rewrite user questions for a retrieval system.
Given the chat history and the latest user message, produce a standalone search query
that can be understood without the history.
If the latest message is already standalone, return it unchanged.
Return only the rewritten question, no preamble."""

ANSWER_SYSTEM = """You are a RAG assistant for UnsaidTalks and any documents the user uploaded.

Rules:
- Answer using the retrieved CONTEXT. Prefer the wording from FAQ "Answer:" fields when present.
- Use CHAT HISTORY so follow-ups (pronouns, "that program", "how much is it") stay consistent.
- If the answer is not in the context, say exactly: I don't know.
- Do not invent fees, names, policies, or program details that are not in the context.
- Mention the source filename when it helps the user trust the answer.

CONTEXT:
{context}"""


def format_history(history: list[dict], limit: int = 12) -> str:
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-limit:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


class ConversationalRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def set_retriever(self, retriever) -> None:
        self.retriever = retriever

    def standalone_question(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        prompt = (
            f"Chat history:\n{format_history(history)}\n\n"
            f"Latest user message: {question}\n\nStandalone search query:"
        )
        rewritten = extract_text(
            self.llm.invoke(
                [SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=prompt)]
            )
        ).strip()
        return rewritten or question

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        standalone = self.standalone_question(question, history)
        docs = self.retriever.invoke(standalone)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}"
            f"{' p.' + str(d.metadata['page']) if d.metadata.get('page') else ''}]\n"
            f"{d.page_content}"
            for d in docs
        ) or "(no retrieved passages)"

        user_block = (
            f"CHAT HISTORY:\n{format_history(history)}\n\n"
            f"QUESTION: {question}\n\n"
            f"(Retrieval query used: {standalone})"
        )
        answer = extract_text(
            self.llm.invoke(
                [
                    SystemMessage(content=ANSWER_SYSTEM.format(context=context)),
                    HumanMessage(content=user_block),
                ]
            )
        ).strip()
        return {
            "answer": answer,
            "source_documents": docs,
            "standalone_question": standalone,
        }

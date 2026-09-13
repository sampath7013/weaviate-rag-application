from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import settings


logger = logging.getLogger(__name__)


client = OpenAI(
    api_key=settings.openai_api_key
)


FALLBACK_ANSWER = (
    "The available document does not contain enough "
    "relevant information to answer this question."
)


def answerability_accuracy(
    expected_answerable: bool,
    generated_answer: str,
) -> float:
    """
    Measure whether the system correctly decided
    to answer or refuse.

    Returns:
        1.0 -> correct behavior
        0.0 -> incorrect behavior
    """

    normalized_answer = generated_answer.strip().lower()
    normalized_fallback = FALLBACK_ANSWER.lower()

    system_answered = (
        normalized_answer != normalized_fallback
    )

    return float(
        system_answered == expected_answerable
    )


def _build_context(
    retrieved_documents: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved chunks into judge-friendly context.
    """

    if not retrieved_documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        document_name = document.get(
            "document_name",
            "Unknown document",
        )

        page_number = document.get(
            "page_number",
            "N/A",
        )

        text = document.get(
            "text",
            "",
        )

        context_parts.append(
            (
                f"[Source {index}]\n"
                f"Document: {document_name}\n"
                f"Page: {page_number}\n"
                f"Content:\n{text}"
            )
        )

    return "\n\n".join(context_parts)


def _run_llm_judge(
    instructions: str,
    input_text: str,
) -> dict[str, Any]:
    """
    Run an OpenAI LLM judge.

    Expected JSON response:
    {
        "score": 0.0-1.0,
        "reason": "..."
    }
    """

    response = client.responses.create(
        model=settings.llm_model,
        instructions=instructions,
        input=input_text,
    )

    raw_output = response.output_text.strip()

    try:
        result = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        logger.warning(
            "generation_metric_invalid_json | output=%s",
            raw_output,
        )

        return {
            "score": 0.0,
            "reason": (
                "The LLM judge returned an invalid "
                "JSON response."
            ),
        }

    score = result.get(
        "score",
        0.0,
    )

    try:
        score = float(score)

    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    return {
        "score": score,
        "reason": result.get(
            "reason",
            "",
        ),
    }


def evaluate_answer_correctness(
    question: str,
    expected_answer: str | None,
    generated_answer: str,
) -> dict[str, Any]:
    """
    Evaluate semantic correctness of the generated answer
    against the expected reference answer.
    """

    if expected_answer is None:

        return {
            "score": 1.0,
            "reason": (
                "No reference answer exists because "
                "the evaluation case is unanswerable."
            ),
        }

    instructions = """
You are evaluating a Retrieval-Augmented Generation system.

Compare the generated answer with the expected reference answer.

Evaluate whether the generated answer contains the same important
facts and meaning as the expected answer.

Do not require exact wording.

Score from 0.0 to 1.0:

1.0 = fully correct
0.75 = mostly correct with minor omissions
0.5 = partially correct
0.25 = mostly incorrect
0.0 = completely incorrect

Return ONLY valid JSON using exactly this structure:

{
  "score": 0.0,
  "reason": "short explanation"
}
""".strip()

    input_text = f"""
QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}
""".strip()

    return _run_llm_judge(
        instructions=instructions,
        input_text=input_text,
    )


def evaluate_groundedness(
    question: str,
    generated_answer: str,
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate whether the generated answer is supported
    by the retrieved document context.
    """

    context = _build_context(
        retrieved_documents
    )

    if not context:

        if (
            generated_answer.strip().lower()
            == FALLBACK_ANSWER.lower()
        ):
            return {
                "score": 1.0,
                "reason": (
                    "The system correctly refused "
                    "because no relevant context "
                    "was available."
                ),
            }

        return {
            "score": 0.0,
            "reason": (
                "The system generated an answer "
                "without retrieved document context."
            ),
        }

    instructions = """
You are evaluating the groundedness of a
Retrieval-Augmented Generation system.

Determine whether every important factual claim in the generated
answer is supported by the supplied document context.

Do not use outside knowledge.

Score from 0.0 to 1.0:

1.0 = fully supported by the context
0.75 = mostly supported
0.5 = partially supported
0.25 = weakly supported
0.0 = unsupported or hallucinated

Return ONLY valid JSON using exactly this structure:

{
  "score": 0.0,
  "reason": "short explanation"
}
""".strip()

    input_text = f"""
QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}
""".strip()

    return _run_llm_judge(
        instructions=instructions,
        input_text=input_text,
    )


def evaluate_citation_correctness(
    expected_pages: list[int],
    retrieved_documents: list[dict[str, Any]],
) -> float:
    """
    Measure whether retrieved source pages include
    the expected supporting pages.

    For now, this is a deterministic page-level metric.

    Returns:
        1.0 -> at least one expected page retrieved
        0.0 -> no expected page retrieved
    """

    if not expected_pages:

        return 1.0

    retrieved_pages = {
        document.get(
            "page_number"
        )
        for document in retrieved_documents
        if document.get(
            "page_number"
        ) is not None
    }

    expected = set(
        expected_pages
    )

    return float(
        bool(
            retrieved_pages.intersection(
                expected
            )
        )
    )


def evaluate_generation(
    question: str,
    expected_answer: str | None,
    expected_pages: list[int],
    expected_answerable: bool,
    generated_answer: str,
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run all generation-related evaluation metrics.
    """

    answerability = answerability_accuracy(
        expected_answerable=expected_answerable,
        generated_answer=generated_answer,
    )

    correctness = evaluate_answer_correctness(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    groundedness = evaluate_groundedness(
        question=question,
        generated_answer=generated_answer,
        retrieved_documents=retrieved_documents,
    )

    citation_score = evaluate_citation_correctness(
        expected_pages=expected_pages,
        retrieved_documents=retrieved_documents,
    )

    return {
        "answerability_accuracy": answerability,
        "answer_correctness": correctness,
        "groundedness": groundedness,
        "citation_correctness": citation_score,
    }
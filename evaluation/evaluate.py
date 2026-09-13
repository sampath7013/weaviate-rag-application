from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from weaviate.exceptions import WeaviateGRPCUnavailableError


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Application imports
# ============================================================

from app.generation.rag_chain import ask_rag
from evaluation.generation_metrics import evaluate_generation
from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset.json"
)

TOP_K = 5
CANDIDATE_K = 10
ALPHA = 0.5
MIN_SCORE = 0.2

MAX_RETRY_ATTEMPTS = 3


# ============================================================
# Dataset loading
# ============================================================

def load_dataset() -> list[dict[str, Any]]:
    """
    Load evaluation/dataset.json.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "dataset.json must contain a JSON array."
        )

    return dataset


# ============================================================
# Helpers
# ============================================================

def extract_retrieved_pages(
    sources: list[dict[str, Any]],
) -> list[int]:
    """
    Extract retrieved page numbers while preserving order.
    """

    return [
        source["page_number"]
        for source in sources
        if source.get("page_number") is not None
    ]


def average(
    values: list[float],
) -> float:
    """
    Safely calculate average.
    """

    if not values:
        return 0.0

    return mean(values)


# ============================================================
# RAG execution with retry
# ============================================================

def run_rag_with_retry(
    *,
    question: str,
    document_id: str | None,
    rerank: bool,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    """
    Execute the RAG pipeline.

    Retries transient Weaviate gRPC connection failures
    using exponential backoff.

    Example waits:

        attempt 1 failure -> wait 2 seconds
        attempt 2 failure -> wait 4 seconds
        attempt 3 failure -> raise error
    """

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            return ask_rag(
                question=question,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
                document_id=document_id,
                alpha=ALPHA,
                min_score=MIN_SCORE,
                rerank=rerank,
            )

        except WeaviateGRPCUnavailableError:

            if attempt == max_attempts:

                logger.exception(
                    (
                        "weaviate_retry_exhausted | "
                        "attempt=%s/%s | "
                        "rerank=%s | "
                        "question=%s"
                    ),
                    attempt,
                    max_attempts,
                    rerank,
                    question,
                )

                raise

            wait_seconds = 2 ** attempt

            logger.warning(
                (
                    "weaviate_retry | "
                    "attempt=%s/%s | "
                    "wait_seconds=%s | "
                    "rerank=%s | "
                    "question=%s"
                ),
                attempt,
                max_attempts,
                wait_seconds,
                rerank,
                question,
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "RAG execution failed after retry attempts."
    )


# ============================================================
# Evaluate one case
# ============================================================

def evaluate_case(
    case: dict[str, Any],
    document_id: str | None,
    rerank: bool,
) -> dict[str, Any]:
    """
    Evaluate one dataset question.

    rerank=False:
        Hybrid retrieval baseline

    rerank=True:
        Hybrid retrieval + reranking
    """

    question_id = case["id"]
    question = case["question"]

    expected_answer = case.get(
        "expected_answer"
    )

    expected_pages = case.get(
        "expected_pages",
        [],
    )

    expected_answerable = case.get(
        "answerable",
        True,
    )

    logger.info(
        (
            "evaluation_case_started | "
            "id=%s | "
            "rerank=%s | "
            "question=%s"
        ),
        question_id,
        rerank,
        question,
    )


    # ========================================================
    # Run actual RAG pipeline
    # ========================================================

    rag_result = run_rag_with_retry(
        question=question,
        document_id=document_id,
        rerank=rerank,
    )

    generated_answer = rag_result.get(
        "answer",
        "",
    )

    sources = rag_result.get(
        "sources",
        [],
    )

    retrieved_pages = extract_retrieved_pages(
        sources
    )


    # ========================================================
    # Retrieval metrics
    # ========================================================

    if expected_answerable:

        hit_rate = hit_rate_at_k(
            retrieved_pages=retrieved_pages,
            expected_pages=expected_pages,
            k=TOP_K,
        )

        recall = recall_at_k(
            retrieved_pages=retrieved_pages,
            expected_pages=expected_pages,
            k=TOP_K,
        )

        rr = reciprocal_rank(
            retrieved_pages=retrieved_pages,
            expected_pages=expected_pages,
        )

    else:

        hit_rate = None
        recall = None
        rr = None


    # ========================================================
    # Generation metrics
    # ========================================================

    generation = evaluate_generation(
        question=question,
        expected_answer=expected_answer,
        expected_pages=expected_pages,
        expected_answerable=expected_answerable,
        generated_answer=generated_answer,
        retrieved_documents=sources,
    )


    # ========================================================
    # Final result
    # ========================================================

    result = {
        "id": question_id,
        "question": question,
        "expected_answerable": expected_answerable,
        "expected_pages": expected_pages,
        "retrieved_pages": retrieved_pages,
        "generated_answer": generated_answer,
        "source_count": len(sources),
        "reranked": rerank,
        "retrieval": {
            "hit_rate_at_k": hit_rate,
            "recall_at_k": recall,
            "reciprocal_rank": rr,
        },
        "generation": generation,
    }

    logger.info(
        (
            "evaluation_case_completed | "
            "id=%s | "
            "rerank=%s"
        ),
        question_id,
        rerank,
    )

    return result


# ============================================================
# Aggregate metrics
# ============================================================

def calculate_summary(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Calculate aggregate evaluation metrics.

    Retrieval metrics are calculated only across
    answerable questions.
    """

    positive_results = [
        result
        for result in results
        if result["expected_answerable"]
    ]


    hit_rates = [
        result["retrieval"]["hit_rate_at_k"]
        for result in positive_results
        if result["retrieval"]["hit_rate_at_k"]
        is not None
    ]


    recalls = [
        result["retrieval"]["recall_at_k"]
        for result in positive_results
        if result["retrieval"]["recall_at_k"]
        is not None
    ]


    reciprocal_ranks = [
        result["retrieval"]["reciprocal_rank"]
        for result in positive_results
        if result["retrieval"]["reciprocal_rank"]
        is not None
    ]


    answerability_scores = [
        result["generation"][
            "answerability_accuracy"
        ]
        for result in results
    ]


    correctness_scores = [
        result["generation"][
            "answer_correctness"
        ][
            "score"
        ]
        for result in positive_results
    ]


    groundedness_scores = [
        result["generation"][
            "groundedness"
        ][
            "score"
        ]
        for result in results
    ]


    citation_scores = [
        result["generation"][
            "citation_correctness"
        ]
        for result in positive_results
    ]


    return {
        "hit_rate_at_k": average(
            hit_rates
        ),
        "recall_at_k": average(
            recalls
        ),
        "mrr": average(
            reciprocal_ranks
        ),
        "answerability_accuracy": average(
            answerability_scores
        ),
        "answer_correctness": average(
            correctness_scores
        ),
        "groundedness": average(
            groundedness_scores
        ),
        "citation_correctness": average(
            citation_scores
        ),
    }


# ============================================================
# Run one complete evaluation mode
# ============================================================

def run_evaluation(
    dataset: list[dict[str, Any]],
    document_id: str | None,
    rerank: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
]:
    """
    Run all evaluation cases.

    Returns:
        individual results
        aggregate summary
    """

    results = []

    for case in dataset:

        try:

            result = evaluate_case(
                case=case,
                document_id=document_id,
                rerank=rerank,
            )

            results.append(
                result
            )

        except Exception:

            logger.exception(
                (
                    "evaluation_case_failed | "
                    "id=%s | "
                    "rerank=%s"
                ),
                case.get(
                    "id",
                    "unknown",
                ),
                rerank,
            )

    summary = calculate_summary(
        results
    )

    return (
        results,
        summary,
    )


# ============================================================
# Aggregate comparison
# ============================================================

def print_comparison(
    baseline: dict[str, float],
    reranked: dict[str, float],
) -> None:
    """
    Print overall baseline vs reranking metrics.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RAG EVALUATION COMPARISON"
    )

    print(
        "=" * 78
    )

    print(
        f"{'Metric':<30}"
        f"{'Baseline':>15}"
        f"{'Reranked':>15}"
        f"{'Change':>15}"
    )

    print(
        "-" * 78
    )


    metric_names = [
        (
            f"Hit Rate@{TOP_K}",
            "hit_rate_at_k",
        ),
        (
            f"Recall@{TOP_K}",
            "recall_at_k",
        ),
        (
            "MRR",
            "mrr",
        ),
        (
            "Answerability Accuracy",
            "answerability_accuracy",
        ),
        (
            "Answer Correctness",
            "answer_correctness",
        ),
        (
            "Groundedness",
            "groundedness",
        ),
        (
            "Citation Correctness",
            "citation_correctness",
        ),
    ]


    for display_name, key in metric_names:

        baseline_value = baseline[
            key
        ]

        reranked_value = reranked[
            key
        ]

        change = (
            reranked_value
            - baseline_value
        )

        print(
            f"{display_name:<30}"
            f"{baseline_value:>15.3f}"
            f"{reranked_value:>15.3f}"
            f"{change:>+15.3f}"
        )


    print(
        "=" * 78
    )


# ============================================================
# Per-question comparison
# ============================================================

def print_case_comparison(
    baseline_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> None:
    """
    Show exactly how reranking changes each question.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PER-QUESTION RETRIEVAL COMPARISON"
    )

    print(
        "=" * 100
    )


    reranked_by_id = {
        result["id"]: result
        for result in reranked_results
    }


    for baseline in baseline_results:

        reranked = reranked_by_id.get(
            baseline["id"]
        )

        if not reranked:
            continue


        print(
            f"\n{baseline['id']}: "
            f"{baseline['question']}"
        )


        print(
            "Expected answerable:",
            baseline[
                "expected_answerable"
            ],
        )

        print(
            "Expected pages    :",
            baseline[
                "expected_pages"
            ],
        )

        print(
            "Baseline pages    :",
            baseline[
                "retrieved_pages"
            ],
        )

        print(
            "Reranked pages    :",
            reranked[
                "retrieved_pages"
            ],
        )


        # ====================================================
        # Positive cases
        # ====================================================

        if baseline[
            "expected_answerable"
        ]:

            baseline_hit = (
                baseline[
                    "retrieval"
                ][
                    "hit_rate_at_k"
                ]
            )

            reranked_hit = (
                reranked[
                    "retrieval"
                ][
                    "hit_rate_at_k"
                ]
            )


            baseline_recall = (
                baseline[
                    "retrieval"
                ][
                    "recall_at_k"
                ]
            )

            reranked_recall = (
                reranked[
                    "retrieval"
                ][
                    "recall_at_k"
                ]
            )


            baseline_rr = (
                baseline[
                    "retrieval"
                ][
                    "reciprocal_rank"
                ]
            )

            reranked_rr = (
                reranked[
                    "retrieval"
                ][
                    "reciprocal_rank"
                ]
            )


            baseline_correctness = (
                baseline[
                    "generation"
                ][
                    "answer_correctness"
                ][
                    "score"
                ]
            )

            reranked_correctness = (
                reranked[
                    "generation"
                ][
                    "answer_correctness"
                ][
                    "score"
                ]
            )


            baseline_groundedness = (
                baseline[
                    "generation"
                ][
                    "groundedness"
                ][
                    "score"
                ]
            )

            reranked_groundedness = (
                reranked[
                    "generation"
                ][
                    "groundedness"
                ][
                    "score"
                ]
            )


            baseline_citation = (
                baseline[
                    "generation"
                ][
                    "citation_correctness"
                ]
            )

            reranked_citation = (
                reranked[
                    "generation"
                ][
                    "citation_correctness"
                ]
            )


            print(
                f"Hit Rate@{TOP_K:<6}: "
                f"{baseline_hit:.3f} "
                f"-> "
                f"{reranked_hit:.3f}"
            )

            print(
                f"Recall@{TOP_K:<8}: "
                f"{baseline_recall:.3f} "
                f"-> "
                f"{reranked_recall:.3f}"
            )

            print(
                "Reciprocal Rank : "
                f"{baseline_rr:.3f} "
                f"-> "
                f"{reranked_rr:.3f}"
            )

            print(
                "Correctness      : "
                f"{baseline_correctness:.3f} "
                f"-> "
                f"{reranked_correctness:.3f}"
            )

            print(
                "Groundedness     : "
                f"{baseline_groundedness:.3f} "
                f"-> "
                f"{reranked_groundedness:.3f}"
            )

            print(
                "Citation         : "
                f"{baseline_citation:.3f} "
                f"-> "
                f"{reranked_citation:.3f}"
            )


        # ====================================================
        # Negative cases
        # ====================================================

        else:

            baseline_answerability = (
                baseline[
                    "generation"
                ][
                    "answerability_accuracy"
                ]
            )

            reranked_answerability = (
                reranked[
                    "generation"
                ][
                    "answerability_accuracy"
                ]
            )


            print(
                "Answerability    : "
                f"{baseline_answerability:.3f} "
                f"-> "
                f"{reranked_answerability:.3f}"
            )


    print(
        "\n"
        + "=" * 100
    )


# ============================================================
# Completion validation
# ============================================================

def validate_complete_evaluation(
    *,
    dataset: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> bool:
    """
    Ensure both evaluation modes completed exactly
    the same full dataset.

    Prevents misleading comparisons when cloud/network
    failures cause cases to be skipped.
    """

    expected_ids = {
        case["id"]
        for case in dataset
    }

    baseline_ids = {
        result["id"]
        for result in baseline_results
    }

    reranked_ids = {
        result["id"]
        for result in reranked_results
    }


    baseline_missing = (
        expected_ids
        - baseline_ids
    )

    reranked_missing = (
        expected_ids
        - reranked_ids
    )


    complete = (
        not baseline_missing
        and not reranked_missing
        and len(baseline_results) == len(dataset)
        and len(reranked_results) == len(dataset)
    )


    if complete:

        print(
            "\nEvaluation completeness check:"
        )

        print(
            f"  Expected cases : {len(dataset)}"
        )

        print(
            f"  Baseline cases : {len(baseline_results)}"
        )

        print(
            f"  Reranked cases : {len(reranked_results)}"
        )

        print(
            "  Status         : COMPLETE"
        )

        return True


    print(
        "\n"
        + "!" * 78
    )

    print(
        "WARNING: EVALUATION IS INCOMPLETE"
    )

    print(
        "!" * 78
    )


    print(
        f"Expected cases : {len(dataset)}"
    )

    print(
        f"Baseline cases : {len(baseline_results)}"
    )

    print(
        f"Reranked cases : {len(reranked_results)}"
    )


    if baseline_missing:

        print(
            "Missing baseline IDs:",
            sorted(
                baseline_missing
            ),
        )


    if reranked_missing:

        print(
            "Missing reranked IDs:",
            sorted(
                reranked_missing
            ),
        )


    print(
        "\nDo NOT use aggregate metrics from this run "
        "for baseline-vs-reranking conclusions."
    )

    print(
        "Retry the evaluation until both modes "
        "complete every dataset case."
    )

    print(
        "!" * 78
    )

    return False


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Compare:

        Hybrid retrieval baseline

    versus:

        Hybrid retrieval + LLM reranking
    """

    dataset = load_dataset()


    print(
        "\nStarting baseline vs reranking evaluation..."
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Cases: {len(dataset)}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    print(
        f"Candidate K: {CANDIDATE_K}"
    )

    print(
        f"Hybrid alpha: {ALPHA}"
    )

    print(
        f"Minimum score: {MIN_SCORE}"
    )

    print(
        f"Weaviate retry attempts: {MAX_RETRY_ATTEMPTS}"
    )


    # ========================================================
    # Document filter
    # ========================================================

    document_id = input(
        "\nEnter the document_id for the indexed "
        "test PDF: "
    ).strip()

    if not document_id:
        document_id = None


    # ========================================================
    # Baseline evaluation
    # ========================================================

    print(
        "\nRunning BASELINE evaluation..."
    )

    baseline_results, baseline_summary = (
        run_evaluation(
            dataset=dataset,
            document_id=document_id,
            rerank=False,
        )
    )


    # ========================================================
    # Reranked evaluation
    # ========================================================

    print(
        "\nRunning RERANKED evaluation..."
    )

    reranked_results, reranked_summary = (
        run_evaluation(
            dataset=dataset,
            document_id=document_id,
            rerank=True,
        )
    )


    # ========================================================
    # Validate experiment
    # ========================================================

    complete = validate_complete_evaluation(
        dataset=dataset,
        baseline_results=baseline_results,
        reranked_results=reranked_results,
    )


    if not complete:
        return


    # ========================================================
    # Aggregate comparison
    # ========================================================

    print_comparison(
        baseline=baseline_summary,
        reranked=reranked_summary,
    )


    # ========================================================
    # Per-question comparison
    # ========================================================

    print_case_comparison(
        baseline_results=baseline_results,
        reranked_results=reranked_results,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app.generation.rag_chain import ask_rag


DATASET_PATH = Path(
    __file__
).parent / "dataset.json"


def load_dataset() -> list[dict[str, Any]]:
    """
    Load the frozen evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data


def _valid_scores(
    scores: list[Any],
) -> list[float]:
    """
    Keep only numeric retrieval scores.
    """

    return [
        float(score)
        for score in scores
        if isinstance(
            score,
            (int, float),
        )
    ]


def _score_statistics(
    scores: list[Any],
) -> dict[str, float | None]:
    """
    Calculate retrieval score diagnostics.

    Returns:
        max score
        min score
        mean score
        top-1/top-2 gap
    """

    valid_scores = _valid_scores(
        scores
    )

    if not valid_scores:

        return {
            "max_retrieval_score": None,
            "min_retrieval_score": None,
            "mean_retrieval_score": None,
            "score_gap_top1_top2": None,
        }

    sorted_scores = sorted(
        valid_scores,
        reverse=True,
    )

    max_score = sorted_scores[0]

    min_score = sorted_scores[-1]

    mean_score = statistics.mean(
        sorted_scores
    )

    if len(sorted_scores) >= 2:

        score_gap = (
            sorted_scores[0]
            - sorted_scores[1]
        )

    else:

        score_gap = None

    return {
        "max_retrieval_score": max_score,
        "min_retrieval_score": min_score,
        "mean_retrieval_score": mean_score,
        "score_gap_top1_top2": score_gap,
    }


def evaluate_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one evaluation case and collect
    retrieval/relevance diagnostics.
    """

    question = case["question"]

    document_id = case.get(
        "document_id"
    )

    answerable = case.get(
        "answerable"
    )

    print(
        f"\nQuestion: {question}"
    )

    result = ask_rag(
        question=question,
        document_id=document_id,
        top_k=5,
        alpha=0.5,
        min_score=0.2,
        rerank=False,
    )

    trace = result.get(
        "trace",
        {}
    )

    metadata = trace.get(
        "metadata",
        {}
    )

    stage_durations = trace.get(
        "stage_durations_ms",
        {}
    )

    retrieval_scores = metadata.get(
        "retrieval_scores",
        [],
    )

    stats = _score_statistics(
        retrieval_scores
    )

    diagnostic = {
        "id": case.get(
            "id"
        ),

        "question": question,

        "answerable": answerable,

        "answer": result.get(
            "answer"
        ),

        "source_count": len(
            result.get(
                "sources",
                [],
            )
        ),

        "retrieval_scores": (
            retrieval_scores
        ),

        **stats,

        "retrieved_pages": (
            metadata.get(
                "retrieved_pages",
                [],
            )
        ),

        "relevance_gate_called": (
            metadata.get(
                "relevance_gate_called"
            )
        ),

        "relevant": metadata.get(
            "relevant"
        ),

        "fallback_reason": (
            metadata.get(
                "fallback_reason"
            )
        ),

        "embedding_cache_hit": (
            metadata.get(
                "embedding_cache_hit"
            )
        ),

        "embedding_ms": (
            stage_durations.get(
                "embedding"
            )
        ),

        "weaviate_search_ms": (
            stage_durations.get(
                "weaviate_search"
            )
        ),

        "relevance_gate_ms": (
            stage_durations.get(
                "relevance_gate"
            )
        ),

        "generation_ms": (
            stage_durations.get(
                "generation"
            )
        ),

        "total_duration_ms": (
            trace.get(
                "total_duration_ms"
            )
        ),
    }

    return diagnostic


def print_case_result(
    result: dict[str, Any],
) -> None:
    """
    Print one diagnostic result.
    """

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        f"ID: {result['id']}"
    )

    print(
        f"Answerable: "
        f"{result['answerable']}"
    )

    print(
        f"Max score: "
        f"{result['max_retrieval_score']}"
    )

    print(
        f"Min score: "
        f"{result['min_retrieval_score']}"
    )

    print(
        f"Mean score: "
        f"{result['mean_retrieval_score']}"
    )

    print(
        f"Top1-Top2 gap: "
        f"{result['score_gap_top1_top2']}"
    )

    print(
        f"Relevance gate called: "
        f"{result['relevance_gate_called']}"
    )

    print(
        f"Relevant: "
        f"{result['relevant']}"
    )

    print(
        f"Fallback reason: "
        f"{result['fallback_reason']}"
    )

    print(
        f"Total latency: "
        f"{result['total_duration_ms']} ms"
    )


def print_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Print answerable vs unanswerable score
    distributions.
    """

    answerable_results = [
        result
        for result in results
        if result.get(
            "answerable"
        ) is True
    ]

    unanswerable_results = [
        result
        for result in results
        if result.get(
            "answerable"
        ) is False
    ]


    print(
        "\n\n"
        "========================================"
    )

    print(
        "RETRIEVAL DIAGNOSTIC SUMMARY"
    )

    print(
        "========================================"
    )


    _print_group_summary(
        "ANSWERABLE",
        answerable_results,
    )

    _print_group_summary(
        "UNANSWERABLE",
        unanswerable_results,
    )


def _print_group_summary(
    name: str,
    results: list[dict[str, Any]],
) -> None:

    max_scores = [
        result[
            "max_retrieval_score"
        ]
        for result in results
        if result.get(
            "max_retrieval_score"
        ) is not None
    ]

    mean_scores = [
        result[
            "mean_retrieval_score"
        ]
        for result in results
        if result.get(
            "mean_retrieval_score"
        ) is not None
    ]

    score_gaps = [
        result[
            "score_gap_top1_top2"
        ]
        for result in results
        if result.get(
            "score_gap_top1_top2"
        ) is not None
    ]


    print(
        f"\n{name}"
    )

    print(
        f"Cases: {len(results)}"
    )


    if max_scores:

        print(
            "Max retrieval score:"
        )

        print(
            f"  minimum: "
            f"{min(max_scores):.4f}"
        )

        print(
            f"  maximum: "
            f"{max(max_scores):.4f}"
        )

        print(
            f"  average: "
            f"{statistics.mean(max_scores):.4f}"
        )


    if mean_scores:

        print(
            "Mean retrieval score:"
        )

        print(
            f"  minimum: "
            f"{min(mean_scores):.4f}"
        )

        print(
            f"  maximum: "
            f"{max(mean_scores):.4f}"
        )

        print(
            f"  average: "
            f"{statistics.mean(mean_scores):.4f}"
        )


    if score_gaps:

        print(
            "Top1-Top2 score gap:"
        )

        print(
            f"  minimum: "
            f"{min(score_gaps):.4f}"
        )

        print(
            f"  maximum: "
            f"{max(score_gaps):.4f}"
        )

        print(
            f"  average: "
            f"{statistics.mean(score_gaps):.4f}"
        )


def save_results(
    results: list[dict[str, Any]],
) -> None:
    """
    Save raw diagnostic results for later analysis.
    """

    output_path = (
        Path(__file__).parent
        / "retrieval_diagnostics_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved diagnostics to: "
        f"{output_path}"
    )


def main() -> None:

    dataset = load_dataset()

    results: list[
        dict[str, Any]
    ] = []


    print(
        "Running retrieval diagnostics..."
    )

    print(
        f"Cases: {len(dataset)}"
    )


    for index, case in enumerate(
        dataset,
        start=1,
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"CASE {index}/{len(dataset)}"
        )

        print(
            "========================================"
        )


        try:

            result = evaluate_case(
                case
            )

            results.append(
                result
            )

            print_case_result(
                result
            )


        except Exception as error:

            print(
                f"\nFAILED: "
                f"{type(error).__name__}: "
                f"{error}"
            )


    print_summary(
        results
    )

    save_results(
        results
    )


if __name__ == "__main__":
    main()
    
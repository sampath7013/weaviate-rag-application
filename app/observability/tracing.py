from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class RAGTrace:
    """
    Lightweight tracing object for one RAG request.

    Tracks:
    - request ID
    - total request latency
    - individual stage latencies
    - metadata
    """

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    started_at: float = field(
        default_factory=time.perf_counter
    )

    stage_started_at: dict[str, float] = field(
        default_factory=dict
    )

    stage_durations_ms: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


    def start_stage(
        self,
        stage_name: str,
    ) -> None:
        """
        Start timing a pipeline stage.
        """

        self.stage_started_at[
            stage_name
        ] = time.perf_counter()

        logger.info(
            (
                "trace_stage_started | "
                "request_id=%s | "
                "stage=%s"
            ),
            self.request_id,
            stage_name,
        )


    def end_stage(
        self,
        stage_name: str,
        **metadata: Any,
    ) -> float:
        """
        Stop timing a pipeline stage.

        Returns:
            duration in milliseconds
        """

        started_at = self.stage_started_at.get(
            stage_name
        )

        if started_at is None:

            logger.warning(
                (
                    "trace_stage_missing_start | "
                    "request_id=%s | "
                    "stage=%s"
                ),
                self.request_id,
                stage_name,
            )

            return 0.0


        duration_ms = (
            time.perf_counter()
            - started_at
        ) * 1000


        self.stage_durations_ms[
            stage_name
        ] = duration_ms


        if metadata:

            self.metadata.update(
                {
                    f"{stage_name}_{key}": value
                    for key, value in metadata.items()
                }
            )


        logger.info(
            (
                "trace_stage_completed | "
                "request_id=%s | "
                "stage=%s | "
                "duration_ms=%.2f | "
                "metadata=%s"
            ),
            self.request_id,
            stage_name,
            duration_ms,
            metadata or {},
        )


        return duration_ms


    def add_metadata(
        self,
        **metadata: Any,
    ) -> None:
        """
        Add request-level metadata.
        """

        self.metadata.update(
            metadata
        )


    def total_duration_ms(
        self,
    ) -> float:
        """
        Return total request duration.
        """

        return (
            time.perf_counter()
            - self.started_at
        ) * 1000


    def complete(
        self,
        *,
        success: bool = True,
    ) -> dict[str, Any]:
        """
        Finalize the trace and return structured trace data.
        """

        total_ms = self.total_duration_ms()


        trace_data = {
            "request_id": self.request_id,
            "success": success,
            "total_duration_ms": round(
                total_ms,
                2,
            ),
            "stage_durations_ms": {
                key: round(
                    value,
                    2,
                )
                for key, value
                in self.stage_durations_ms.items()
            },
            "metadata": self.metadata,
        }


        logger.info(
            (
                "rag_trace_completed | "
                "request_id=%s | "
                "success=%s | "
                "total_duration_ms=%.2f | "
                "stages=%s | "
                "metadata=%s"
            ),
            self.request_id,
            success,
            total_ms,
            self.stage_durations_ms,
            self.metadata,
        )


        return trace_data


    def fail(
        self,
        error: Exception,
    ) -> dict[str, Any]:
        """
        Record an unsuccessful RAG request.
        """

        self.metadata[
            "error_type"
        ] = type(error).__name__

        self.metadata[
            "error_message"
        ] = str(error)


        logger.exception(
            (
                "rag_trace_failed | "
                "request_id=%s | "
                "error_type=%s | "
                "error=%s"
            ),
            self.request_id,
            type(error).__name__,
            str(error),
        )


        return self.complete(
            success=False
        )
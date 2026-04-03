"""
client.py — Lightweight HTTP client for the ProteinEnv server.

Used by inference.py to talk to the running FastAPI environment
via standard reset / step / state calls.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

try:
    from models import (
        ProteinAction,
        ProteinObservation,
        ProteinState,
        StepResult,
    )
except ImportError:
    from protein_env.models import (
        ProteinAction,
        ProteinObservation,
        ProteinState,
        StepResult,
    )

logger = logging.getLogger(__name__)


class ProteinEnvClient:
    """HTTP client that wraps the ProteinEnv FastAPI server.

    Provides reset() / step() / state() matching the server-side
    ProteinEnvironment API so inference.py can drive the environment
    as if it were in-process.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        """Initialise the client pointing at a running server.

        Args:
            base_url: Root URL of the ProteinEnv server.
            timeout:  Request timeout in seconds.

        Returns:
            None.

        Raises:
            Nothing.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        task_type: str = "easy",
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> ProteinObservation:
        """Start a new episode and return the initial observation.

        Args:
            task_type:  One of 'easy', 'medium', 'hard'. Case-insensitive.
            seed:       Optional random seed for reproducibility.
            episode_id: Optional episode ID override.

        Returns:
            ProteinObservation parsed from the server response.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        payload: dict = {"task_type": task_type}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return ProteinObservation.model_validate(resp.json())

    def step(self, action: ProteinAction) -> StepResult:
        """Execute one action and return the step result.

        Args:
            action: Validated ProteinAction to send to the server.

        Returns:
            StepResult with observation, reward, done flag, and info.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        resp = self._client.post(
            "/step",
            content=action.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> ProteinState:
        """Fetch the current full episode state snapshot.

        Args:
            None.

        Returns:
            ProteinState with all episode tracking fields.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        resp = self._client.get("/state")
        resp.raise_for_status()
        return ProteinState.model_validate(resp.json())

    def health(self) -> dict:
        """Ping the /health endpoint.

        Args:
            None.

        Returns:
            dict with status field.

        Raises:
            httpx.HTTPStatusError: On server error.
        """
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP connection pool.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing.
        """
        self._client.close()

    def __enter__(self) -> "ProteinEnvClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — closes the HTTP client."""
        self.close()

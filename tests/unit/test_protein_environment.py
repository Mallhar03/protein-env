"""
tests/unit/test_protein_environment.py — Coverage booster for Section 3 audit.

Tests the ProteinEnvironment orchestrator, ensuring it correctly wires
StateManager, ESM2Embedder, and RewardCalculator into the OpenEnv loop.
"""

from unittest.mock import MagicMock, patch
import pytest

from server.protein_environment import ProteinEnvironment
from models import TaskType, ProteinObservation, ProteinAction, ActionType


@pytest.fixture
def mock_env():
    """Create a ProteinEnvironment with mocked heavy dependencies."""
    with patch("server.protein_environment.ESM2Embedder") as mock_calc:
        # Prevent actual ESM2 loading
        mock_calc.return_value.embed.return_value = [0.1] * 320
        env = ProteinEnvironment()
        return env


def test_environment_reset(mock_env):
    """Test that reset returns a ProteinObservation."""
    obs = mock_env.reset(task_type=TaskType.EASY, seed=42)
    assert isinstance(obs, ProteinObservation)
    assert obs.task_type == TaskType.EASY
    assert obs.step_number == 0


def test_environment_step_submit(mock_env):
    """Test that step with SUBMIT_PREDICTION returns a StepResult."""
    mock_env.reset(task_type=TaskType.EASY, seed=42)
    
    action = ProteinAction(
        action_type=ActionType.SUBMIT_PREDICTION,
        predicted_family="Globin family"
    )
    result = mock_env.step(action)
    
    assert result.done is True
    assert result.reward >= 0.0
    assert "reward_breakdown" in result.info.model_dump()


def test_environment_step_tool_call(mock_env):
    """Test that step with CALL_TOOL returns a tool result."""
    mock_env.reset(task_type=TaskType.EASY, seed=42)
    
    action = ProteinAction(
        action_type=ActionType.CALL_TOOL,
        tool_name="get_esm2_embedding",
        tool_args={"sequence": "MAGA"}
    )
    result = mock_env.step(action)
    
    assert result.done is False
    assert result.info.tool_result is not None
    assert "embedding" in result.info.tool_result


def test_environment_state_snapshot(mock_env):
    """Test that state() returns the full internal state."""
    mock_env.reset(task_type=TaskType.EASY, seed=42)
    state = mock_env.state()
    assert state.step_number == 0
    assert state.task_type == TaskType.EASY

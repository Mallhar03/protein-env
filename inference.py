"""
inference.py — ProteinEnv baseline inference script.

Hackathon requirement: must be at project root, named exactly inference.py.

Environment variables:
  API_BASE_URL  — LLM API base URL (OpenAI-compatible)
  MODEL_NAME    — Model identifier string
  HF_TOKEN      — Hugging Face / API key

STDOUT FORMAT (mandatory):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import json
import signal
from typing import List, Optional

import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

try:
    from client import ProteinEnvClient
except ImportError:
    ProteinEnvClient = None  # type: ignore

from models import ProteinAction

load_dotenv()

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN")
ENV_BASE_URL  = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK        = "protein-env"
MAX_STEPS        = 10
SUCCESS_THRESHOLD = 0.5   # mean score >= 0.5 considered success

# ── Structured log helpers (evaluator parses stdout) ─────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompts ────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a protein biology expert AI agent. You will be given a protein sequence
and a task. You have access to one tool: get_esm2_embedding, which returns a
320-dimensional embedding vector for any amino acid sequence.

For each task, respond with a JSON object in exactly this format:
{
  "action_type": "submit_prediction" | "call_tool",
  "tool_name": "get_esm2_embedding",         (only if action_type is call_tool)
  "tool_args": {"sequence": "<AA sequence>"}, (only if action_type is call_tool)
  "predicted_family": "<family name>",        (only for easy task)
  "predicted_go_terms": ["GO:XXXXXXX", ...],  (only for medium task)
  "predicted_pathogenicity": "<value>",       (only for hard task)
  "predicted_diseases": ["<disease>", ...],   (only for hard task)
  "reasoning": "<your reasoning>"
}

Valid pathogenicity values: Pathogenic, Likely pathogenic,
Variant of Uncertain Significance, Likely benign, Benign"""

GO_HINT = (
    "\n\nFor GO terms, predict using format GO:XXXXXXX (7 digits).\n"
    "Common molecular function terms: GO:0003700, GO:0005515, GO:0003677\n"
    "Common biological process terms: GO:0006915, GO:0043065, GO:0008150\n"
    "Common cellular component terms: GO:0005634, GO:0005737, GO:0005829"
)


# ── Timeout (Linux/macOS only) ────────────────────────────────────────────────

def _timeout_handler(signum, frame):  # noqa: ANN001
    print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
    sys.exit(1)

if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(18 * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_task(
    openai_client: openai.OpenAI,
    env,
    task: str,
) -> float:
    """Run one episode for *task* and return the final scalar reward."""

    sys_prompt = BASE_SYSTEM_PROMPT
    if task == "medium":
        sys_prompt += GO_HINT

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        obs      = env.reset(task_type=task)
        messages = [{"role": "system", "content": sys_prompt}]
        done     = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            messages.append({
                "role":    "user",
                "content": f"Observation: {obs.model_dump_json()}",
            })

            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            messages.append({"role": "assistant", "content": raw})

            action_dict = json.loads(raw)
            action      = ProteinAction(**action_dict)
            action_str  = action.action_type

            result = env.step(action)
            reward = result.reward or 0.0
            done   = result.done
            error  = getattr(result, "error", None)
            obs    = result.observation

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

        score   = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    # ── Guard: required env vars ──────────────────────────────────────────
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr, flush=True)
        sys.exit(1)

    if ProteinEnvClient is None:
        print("ERROR: client.py not found — cannot connect to ProteinEnv server.", file=sys.stderr, flush=True)
        sys.exit(1)

    openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env           = ProteinEnvClient(base_url=ENV_BASE_URL)

    # ── Health check: fail fast if server is unreachable ─────────────────
    try:
        import httpx
        r = httpx.get(f"{ENV_BASE_URL}/health", timeout=10)
        if r.status_code != 200:
            print(f"ERROR: Server health check failed (HTTP {r.status_code}). Is the server running at {ENV_BASE_URL}?", file=sys.stderr, flush=True)
            sys.exit(1)
        print(f"[DEBUG] Server healthy at {ENV_BASE_URL}", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot reach server at {ENV_BASE_URL}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    tasks  = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        try:
            scores[task] = run_task(openai_client, env, task)
        except Exception as exc:  # noqa: BLE001
            print(f"[DEBUG] Task {task} failed: {exc}", flush=True)
            scores[task] = 0.0

    # ── Human-readable summary table ─────────────────────────────────────
    print("", flush=True)
    print("=" * 46, flush=True)
    print("  ProteinEnv Baseline Inference Results", flush=True)
    print("=" * 46, flush=True)
    for task in tasks:
        s = scores.get(task, 0.0)
        bar = "🟢" if s >= 0.9 else ("🟡" if s >= 0.4 else "🔴")
        print(f"  {task:<8}  score={s:.3f}  {bar}", flush=True)
    print("-" * 46, flush=True)
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Mean      score={mean:.3f}", flush=True)
    print("=" * 46, flush=True)


if __name__ == "__main__":
    main()

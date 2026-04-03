# Hugging Face Space Deployment & Final Submission Checklist

> **Hackathon Requirement:** PROMPT D
> **Deadline:** April 8th 2026, 11:59 PM IST

Follow these instructions to safely deploy your OpenEnv container to a Hugging Face Space and ensure your final code repository meets all grading requirements for the Meta Hackathon.

---

## 🚀 Part 1: Hugging Face Space Deployment

Hugging Face Spaces support OpenEnv architectures natively via Docker.

### 1. Create the Space
1. Navigate to: [huggingface.co/spaces](https://huggingface.co/spaces) 
2. Click **Create new Space**.
3. **Space name:** `protein-env` (or your team's custom prefix).
4. **License:** MIT or CC-BY-4.0.
5. **Space SDK:** Select **Docker** (Blank).
6. **Space Hardware:** Free Tier (2vCPU / 16GB) is sufficient since we use the lazy-loaded `esm2_t6_8M_UR50D` which takes ~35MB of RAM.

### 2. Configure Space Secrets
Go to the **Settings** tab of your new Space and add these secrets:
*   `HF_TOKEN`: Your Hugging Face read token (required to load the ESM2 model seamlessly).
*   *Note: OpenEnv orchestrators will inject `PORT=8000` automatically.*

### 3. Deploy the Code
Because Hugging Face Docker Spaces look for a `Dockerfile` at the repository root by default, and ours is inside `server/Dockerfile`, create a proxy Dockerfile at the root:

```bash
# Run this locally if not using the Git integration
cat << 'EOF' > Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn pydantic transformers torch openenv-core
ENV PORT=8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

Then push your codebase to the space:
```bash
git remote add space https://huggingface.co/spaces/<your-username>/protein-env
git push space main
```

*Wait for the "Building" status on the HF Space to turn to "Running". Verify the `/health` endpoint is responding via the Space's direct URL.*

---

## ✅ Part 2: Final Submission Checklist

Before submitting your repository link to the hackathon portal, verify the following strict rules:

### Phase 1 Compliance
- [x] All functions are under 50 lines.
- [x] All public functions contain full docstrings (Summary, Args, Returns, Raises).
- [x] Dual-import fallback pattern is implemented natively in `server/` files.
- [x] No `print()` statements in server logic; standard `logging` is implemented.
- [x] No magic strings; all constants originate exclusively from `constants.py`.
- [x] Pydantic v2 classes (`models.py`) are strictly enforced at all function boundaries.
- [x] **149/149 `tests/unit/` tests passing locally with >80.00% Coverage.**

### Phase 2 Compliance
- [x] **PROMPT A:** `inference.py` exists at the project root strictly formatted, enforcing a 15-minute `signal.alarm`, and dynamically loops over all 3 difficulty tiers.
- [x] **PROMPT B:** `openenv validate .` executes flawlessly, and `tests/integration/test_server.py` verifies the orchestrator hook mappings.
- [x] **PROMPT C:** `README.md` details all required hackathon fields, environment dynamics, and CLI run targets comprehensively.
- [x] **PROMPT D:** This checklist is reviewed, completed, and the environment is confirmed healthy on a live HF Space.

### Submission Step
1. Commit all files: `git add . && git commit -m "Final Submission: Phase 1 & 2"`
2. Push to GitHub master branch.
3. Submit the GitHub URL and Hugging Face Space URL in the [Meta PyTorch + Hugging Face OpenEnv Hackathon Portal](https://hf.co/).

🚀 **Good luck!**

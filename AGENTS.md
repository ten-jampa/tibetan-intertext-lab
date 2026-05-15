# Repository Guidelines

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues for this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage labels use default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Domain docs use single-context layout with root `CONTEXT.md` and root `docs/adr/`. See `docs/agents/domain.md`.

## Project Structure & Module Organization
- `tibetan_pipeline/`: main Python package (normalization, segmentation, clumping, pseudo-eval, embeddings, pairwise core/adapters, CLI).
- `tibetan_pipeline/segmenters/`: pluggable segmenter interface and engine adapters.
- `intellexus_engine_code/`: Intellexus-specific engine implementations used by adapters.
- `scripts/`: runnable entry points for end-to-end workflows (pipeline run, benchmark, one-file compare, two-text pairwise).
- `tests/`: unit tests (`test_*.py`) covering CLI, normalization, pairwise core/adapters, clumping, and segmenters.
- `tasks/`: lightweight planning and lessons docs.

## Build, Test, and Development Commands
- Install deps: `python -m pip install -r requirements.txt`
- Run all tests: `python -m unittest discover -s tests -v`
- Run segmentation pipeline: `python scripts/run_tibetan_pipeline.py --input <file> --output-dir output/<run_name> --engine botok_ours`
- Run clumped pseudo-eval: `python scripts/run_clumped_segmentation_eval.py --input <file> --output-dir output/<run_name> --engine botok_ours`
- Run multi-engine benchmark: `python scripts/run_engine_benchmarks.py --input <file> --output-dir output/benchmarks --engines botok_ours botok_intellexus regex_intellexus`
- Run two-text pairwise similarity: `python scripts/run_pairwise_text_similarity.py --text-a <file_a> --text-b <file_b> --output-dir output/<run_name>`

## Coding Style & Naming Conventions
- Use Python 3 with 4-space indentation, type hints, and `from __future__ import annotations` in new modules when appropriate.
- Prefer `snake_case` for functions/variables, `PascalCase` for classes, and descriptive module names.
- Keep functions focused and side effects explicit; prefer `pathlib.Path` for filesystem logic.
- Match existing docstring style: short, purpose-first module/class docstrings.

## Testing Guidelines
- Framework: built-in `unittest` (no pytest dependency required).
- Place tests in `tests/` and name files `test_<feature>.py`; test methods should start with `test_`.
- Add or update tests for behavior changes, especially CLI argument wiring, pairwise metrics/ranking semantics, and segmentation boundaries.
- Run `python -m unittest discover -s tests -v` before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes seen in history: `feat:`, `docs:`, `fix:`, `test:`, `refactor:`.
- Keep commits scoped to one logical change and include tests/docs in the same PR when relevant.
- PRs should include: concise summary, why the change is needed, validation steps/commands run, and sample output paths (for script changes).
- For workflow changes, include a small reproducible command example in the PR description.

## Security & Configuration Tips
- Do not commit large input datasets or generated outputs (`data/`, `output/` are typically local artifacts).
- Use local cache paths (for example `.cache/botok/dialect_packs`) and avoid hardcoded absolute machine-specific paths.

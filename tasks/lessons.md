# Lessons

- `pyewts` may fail under pip build isolation even when `setuptools` is present; `python -m pip install --no-build-isolation pyewts` works in this environment.
- Botok defaults to a machine-specific cache under `~/Documents/pybo/dialect_packs`; set a repo-local dialect-pack directory such as `.cache/botok/dialect_packs` to keep the backend reproducible inside the workspace.
- Always check corpus size before launching a non-trivial pipeline job; choose an explicit sample limit for interactive evaluation and reserve full-corpus runs for deliberate batch jobs.
- Boundary scoring based on raw segment spans can be misleading if spans include trailing spaces; trim right-side whitespace before computing boundary positions.
- Do not put pip CLI flags such as `--no-build-isolation` inside the `pip:` requirements list in `environment.yml`; conda writes that list to a requirements file, and pip will reject those flags there.
- Installing `ipykernel` in an environment enables notebook support, but it does not register a visible Jupyter kernel by itself; run `python -m ipykernel install --user --name <env-name>` as an explicit verification step.
- Fresh conda environments with very new `setuptools` can lack `pkg_resources`; if a legacy package like `pyewts` imports it during build, pin `setuptools` to a compatible version in `environment.yml`.
- When caching expensive inference objects, key the cache by heavyweight load-time settings only; mutable runtime knobs like batch size and progress logging should update the existing instance instead of forcing a model reload.
- If a verification command is part of the repo standard, add its tool as a project dev dependency instead of reporting that it is missing from the current local environment.

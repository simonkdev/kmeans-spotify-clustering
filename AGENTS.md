# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: entry point that runs preprocessing, k-means optimization, and writes outputs.
- `src/`: core modules
  - `data_preparation.py`: CSV loading/cleanup and optional export to `tmp/processed_data.csv`
  - `optimization.py`: k-means assignment + centroid update loop
  - `cluster_rating.py`: computes a simple iteration/cluster rating from centroid distances
- `data/`: input dataset(s) (e.g. `data/data.csv`)
- `result/`: generated artifacts (centroids + iteration ratings), grouped by cluster count (e.g. `result/centroids/cluster5/`)
- `tmp/`: scratch outputs (e.g. processed CSV)
- `.devenv/`, `devenv.nix`, `.envrc`: reproducible dev environment (Nix + devenv/direnv)

## Build, Test, and Development Commands

- `direnv allow`: load the repo’s Nix/devenv environment (first time only).
- `devenv shell`: enter a shell with Python + dependencies (numpy/pandas/matplotlib/tqdm).
- `python main.py`: run the full pipeline using constants at the top of `main.py`.
- If you change `CLUSTER_COUNT`, ensure output dirs exist:
  - `mkdir -p result/centroids/cluster<N> result/iteration_ratings/cluster<N>`

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8-ish formatting, `snake_case` for functions/vars, `UPPER_SNAKE_CASE` for constants.
- Prefer pure functions in `src/` (pass data in/out) and keep I/O in `main.py`.
- Avoid committing noisy debug prints; use short, targeted logging while developing.

## Testing Guidelines

- No formal test suite is included yet. New contributions that change math/IO should add tests under `tests/` (recommended: `pytest`).
- Suggested naming: `tests/test_<module>.py`; keep fixtures small (use a tiny synthetic matrix, not `data/data.csv`).

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, plain-English, and action-oriented (e.g. “added preprocessing logic”).
- PRs should include:
  - what changed + why, how to run (`python main.py`), and any output expectations (new files under `result/`)
  - notes on dataset assumptions (columns removed, first-row handling) if you touch preprocessing


# Astar Island

A scaffolded Python app for the Astar Island Viking civilisation prediction challenge.

It gives you:

- A typed API client for the round, budget, simulate, and submit endpoints
- Local caching for round details, observations, and generated predictions
- A baseline heuristic predictor that blends map priors with observed viewport outcomes
- A lightweight cross-seed feature model so observations on one seed can still inform the others
- A Streamlit app for exploring seeds, querying the simulator, previewing predictions, and submitting them
- A small CLI for scripting the same workflow

## Quick start

1. Create a virtual environment.
2. Install the package in editable mode.
3. Add your JWT token to `.env`.
4. Launch the Streamlit app or use the CLI.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
astar-island-app
```

The app will open a local Streamlit server. From there you can:

- Load the active round
- Inspect each seed's initial terrain
- Run viewport queries and cache the results
- Build baseline predictions
- Submit one seed or all seeds

## CLI examples

Fetch the active round and cache its details:

```bash
astar-island pull-round
```

Run a simulation query for seed `0`:

```bash
astar-island simulate --seed 0 --x 10 --y 5 --w 15 --h 15
```

Build predictions for all seeds:

```bash
astar-island build
```

Submit the cached predictions:

```bash
astar-island submit --all-seeds
```

## Project layout

- `src/astar_island/api.py`: API client and HTTP error handling
- `src/astar_island/cache.py`: local JSON cache for round data, observations, and predictions
- `src/astar_island/baseline.py`: heuristic predictor and observation fusion
- `src/astar_island/app.py`: Streamlit UI
- `src/astar_island/cli.py`: CLI for scripted workflows

## Notes

- The baseline is intentionally conservative and always applies a probability floor before normalization.
- Cached data is written to `.data/` by default.
- The app assumes a Bearer token, but you can adapt the client to cookie auth if you prefer.

## Tests

After installing dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

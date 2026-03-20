# Astar Island

A scaffolded Python app for the Astar Island Viking civilisation prediction challenge.

It gives you:

- A typed API client for the round, budget, simulate, and submit endpoints
- Local caching for round details, observations, and generated predictions
- A baseline heuristic predictor that blends map priors with observed viewport outcomes
- A lightweight cross-seed feature model so observations on one seed can still inform the others
- A staged query planner that starts with balanced cross-seed exploration, then infers, then concentrates late queries
- Offline analysis tools to fetch completed-round ground truth, evaluate the model, and tune saved weights
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

Plan and run an automated query batch, then refresh predictions:

```bash
astar-island autoquery --count 8 --build
```

Spend the rest of the live round budget, rebuild, and resubmit:

```bash
astar-island autoquery --use-remaining --replan-every 5 --build --submit
```

Backfill completed-round analysis data for offline evaluation:

```bash
astar-island sync-analysis
```

Evaluate the current saved model weights on cached completed rounds:

```bash
astar-island evaluate
```

Tune model weights on cached completed rounds and save the result:

```bash
astar-island tune --save
```

Submit the cached predictions:

```bash
astar-island submit --all-seeds
```

## Project layout

- `src/astar_island/api.py`: API client and HTTP error handling
- `src/astar_island/cache.py`: local JSON cache for round data, observations, and predictions
- `src/astar_island/baseline.py`: probabilistic predictor and observation fusion
- `src/astar_island/planner.py`: uncertainty-driven query planning and batch execution
- `src/astar_island/analysis.py`: offline scoring, evaluation, and parameter tuning
- `src/astar_island/app.py`: Streamlit UI
- `src/astar_island/cli.py`: CLI for scripted workflows

## Notes

- The baseline is intentionally conservative and always applies a probability floor before normalization.
- Cached data is written to `.data/` by default.
- Tuned model weights, when present, are loaded automatically from `.data/model_params.json`.
- The app assumes a Bearer token, but you can adapt the client to cookie auth if you prefer.

## Tests

After installing dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

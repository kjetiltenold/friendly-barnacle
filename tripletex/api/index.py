"""Vercel serverless entrypoint — re-exports the FastAPI app."""

import sys
from pathlib import Path

# Add the tripletex directory to the Python path so `app` package resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

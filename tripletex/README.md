# Tripletex Agent

Small FastAPI app that accepts a `/solve` request, uses an OpenAI-compatible model to plan the work, and executes Tripletex API calls on behalf of the user.

## Requirements

- Python 3.12 recommended
- An OpenAI-compatible API key
- A Tripletex `base_url` and `session_token` when calling `/solve`

## Environment variables

Create a local `.env` file from `.env.example`.

Required:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`
- `MODEL_NAME` defaults to `gpt-4o`
- `LOG_LEVEL` defaults to `INFO`
- `MAX_AGENT_ITERATIONS` defaults to `15`
- `SOFT_TIMEOUT_SECONDS` defaults to `270`

## Run locally on macOS

1. Open Terminal and go to the project folder.

```bash
cd /path/to/tripletex
```

2. Create and activate a virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If `python3.12` is not available, try `python3`.

3. Install dependencies.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Create `.env` from the example file.

```bash
cp .env.example .env
```

5. Edit `.env` and set at least:

```env
OPENAI_API_KEY=sk-...
```

6. Start the API with auto-reload.

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

7. In a new terminal, test the health endpoint.

```bash
curl http://127.0.0.1:8000/health
```

## Run locally on Windows

These steps use PowerShell.

1. Open PowerShell and go to the project folder.

```powershell
cd C:\path\to\tripletex
```

2. Create and activate a virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py -3.12` is not available, try `py`.

3. Install dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Create `.env` from the example file.

```powershell
Copy-Item .env.example .env
```

5. Edit `.env` and set at least:

```env
OPENAI_API_KEY=sk-...
```

6. Start the API with auto-reload.

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

7. Test the health endpoint.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Call `/solve`

`/solve` expects Tripletex credentials in the request body. They are not read from `.env`.

Example request body:

```json
{
  "prompt": "Create a draft expense reimbursement for the attached receipt.",
  "files": [],
  "tripletex_credentials": {
    "base_url": "https://tripletex.no",
    "session_token": "your-session-token"
  }
}
```

### macOS example

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a draft expense reimbursement for the attached receipt.",
    "files": [],
    "tripletex_credentials": {
      "base_url": "https://tripletex.no",
      "session_token": "your-session-token"
    }
  }'
```

### Windows PowerShell example

```powershell
$body = @{
  prompt = "Create a draft expense reimbursement for the attached receipt."
  files = @()
  tripletex_credentials = @{
    base_url = "https://tripletex.no"
    session_token = "your-session-token"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/solve `
  -ContentType "application/json" `
  -Body $body
```

## Run with Docker

If you prefer Docker instead of a virtual environment:

```bash
docker build -t tripletex-agent .
docker run --rm -p 8080:8080 --env-file .env tripletex-agent
```

Then open:

- `http://127.0.0.1:8080/health`
- `http://127.0.0.1:8080/solve`

## Notes

- The app entrypoint for local development is `app.main:app`.
- The Vercel serverless entrypoint is `api/index.py`.
- The app logs `/solve` requests and responses to stdout, which is useful both locally and on Vercel.

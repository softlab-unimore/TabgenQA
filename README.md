# TabQA Generator

A web-based demo for generating tabular Question-Answering benchmarks, powered by [Gradino](https://github.com/softlab-unimore/Gradino). Built for the CIKM 2025 conference demo track.

## Features

- **Parameter configuration** — set domain, question type, table count, sample count, cardinality, and column count via a clean UI
- **Live progress** — real-time progress bar and status updates during generation
- **Stop control** — cancel a running generation at any time
- **Instance editor** — edit questions, answers, and tables (cell values, hierarchical headers, schema changes, add/remove rows & columns)
- **Export** — download the dataset as CSV or JSON

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- An **OpenAI API key** (Gradino uses GPT for table and question generation)

### Run

```bash
git clone https://github.com/lucacontalbo/TabQAGenerator.git
cd TabQAGenerator

# Optional: set API key via .env (or enter it in the UI at runtime)
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

docker compose up --build
```

Then open **http://localhost:8000** in your browser.

> The first build clones Gradino and installs all dependencies (including z3-solver). This may take 3–5 minutes. Subsequent starts are fast.

## Usage

1. **Enter your OpenAI API key** in the top-right field (if not set via `.env`)
2. **Configure parameters** in the left panel:
   | Parameter | Description |
   |-----------|-------------|
   | Domain | Domain for few-shot table examples (environmental / finance / healthcare / products) |
   | Question Type | Aggregation type: sum, average, or superlative |
   | Number of Tables | Tables per instance; set to `-1` for ablation mode (2,3,5,10,20) |
   | Num Samples | How many instances to generate |
   | Col. Cardinality | Cardinality of the latent relational source |
   | Num Columns | Number of columns in the latent relational source |
   | Sequential | Enable multi-hop questions across tables via foreign keys |
3. Click **Generate Benchmark**
4. Watch the progress bar; click **Stop** to cancel at any time
5. Once complete, click the **pencil icon** on any instance to edit:
   - **Question & Answer tab** — edit the question text and expected answer
   - **Tables tab** — full table editor: click cells to select, double-click to edit, use the toolbar to add/remove rows and columns, adjust colspan/rowspan for hierarchical headers
6. **Download** the final dataset as CSV or JSON

## Architecture

```
TabQAGenerator/
├── docker-compose.yml       # Single-service Docker Compose config
├── Dockerfile               # Clones Gradino + installs deps
├── backend/
│   ├── app.py               # FastAPI application (API + static serving)
│   ├── generate_script.py   # Subprocess worker; patches tqdm for progress
│   ├── requirements.txt     # FastAPI + uvicorn
│   └── static/
│       └── index.html       # React + Tailwind single-page app (CDN, no build step)
└── output/                  # (created at runtime) mounted volume for output files
```

**Generation flow:**

1. The React frontend sends a `POST /api/generate` with the chosen parameters
2. The FastAPI backend spawns `generate_script.py` as an async subprocess
3. `generate_script.py` monkey-patches `tqdm` before importing Gradino, so each completed sample emits a JSON progress line to stdout
4. The backend streams these progress events to the frontend via **Server-Sent Events** (`GET /api/generate/{id}/stream`)
5. On completion, the serialised DataFrames are returned as JSON, flattened into a list of instances, and stored in memory
6. The frontend fetches and renders the instances; edits are pushed back via `PUT /api/tasks/{id}/instances/{iid}`

## Notes

- Gradino is cloned **read-only** inside the Docker image; no files in the Gradino repository are modified.
- API keys entered in the UI are sent to the backend over localhost only and are never persisted to disk.
- The in-memory task store is cleared when the container restarts; download your dataset before stopping the container.

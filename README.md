# DocBot

![CI](https://github.com/Jihanvall/ask-my-docs/actions/workflows/ci.yml/badge.svg)

Chat with your own documents. Attach a `.txt`, `.md` or text-based `.pdf` file with the **+** button, ask questions, and get answers that show the passages they came from. If the document does not contain the answer, the assistant says so instead of guessing.

![DocBot screenshot](docs/screenshot.png)

The repository is named `ask-my-docs`; the app is called DocBot.

## Features

- Chat interface with a **+** button to attach a document, or drag and drop a file anywhere on the page.
- Three search methods, chosen from a menu: keywords (TF-IDF), meaning (Gemini embeddings) and a hybrid of both.
- Every answer cites numbered passages, and each passage can be opened to read the original text with a relevance bar.
- A clear "I could not find this" message when the document has no answer.
- Loading states for indexing and answering, a copy button, and a layout that works on phones.
- Measured quality: an evaluation set, a retrieval comparison and cached embeddings.
- Follow-up questions work: the chat remembers the last messages and rewrites a follow-up such as "and then?" into a standalone question before searching.

## How it works

1. **Loading** (`src/loader.py`): reads the file and extracts its text. Scanned PDFs without readable text are rejected with a clear message.
2. **Chunking** (`src/chunker.py`): splits the text into overlapping chunks of words, each keeping its source file and position.
3. **Retrieval**: finds the chunks closest to the question. Three methods are available:
   - TF-IDF (`src/retriever.py`): matches words, runs locally.
   - Embeddings (`src/embedding_retriever.py`): matches meaning using Gemini embeddings, with an on-disk cache (`src/embedding_cache.py`).
   - Hybrid (`src/hybrid_retriever.py`): combines both rankings with reciprocal rank fusion.
4. **Answering** (`src/answerer.py`): sends the question and the top chunks to Gemini, which must answer only from them and cite the excerpts. With no relevant chunk, it answers "I don't know" without calling Gemini.
5. **API and interface**: a FastAPI backend (`api.py`) and a React chat page (`frontend/`).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Gemini API key (required) |
| `MAX_UPLOAD_MB` | `10` | Maximum size of an uploaded file |
| `SESSION_TTL_SECONDS` | `3600` | How long an indexed document stays in memory |
| `MAX_SESSIONS` | `20` | Maximum documents kept in memory, oldest removed first |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Allowed CORS origins |

## Usage

Start the backend:

```
uvicorn api:app --reload
```

Start the frontend in a second terminal:

```
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, press **+** to attach a document, then ask your questions. The search method is chosen from the menu in the top bar and applies to the next document you attach. Keywords are instant; meaning and hybrid call Gemini for every part of the document, so long files can take a few minutes on the free tier the first time.

The API documentation is at `http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Status and number of indexed documents |
| POST | `/documents` | Uploads a file and indexes it with the chosen `method` (`tfidf`, `embedding` or `hybrid`), returns a `session_id` |
| POST | `/ask` | Answers a question for a `session_id`; returns the answer, its sources, the retrieved `passages` and a `refused` flag |

Unsupported files are rejected with 400, files over the size limit with 413, unknown sessions with 404, and Gemini failures with 502.

## Evaluation

The evaluation scripts take a document and a question file, so the same pipeline can be measured on different texts:

```
python run_eval.py --doc data/readme.md --questions eval/questions.json
python run_eval.py --method embedding --doc data/sapiens_part.txt --questions eval/sapiens_questions.json
python compare_retrievers.py --doc data/sapiens_part.txt --questions eval/sapiens_questions.json
```

`--method` chooses the retrieval method: `tfidf` (default), `embedding` or `hybrid`.

### Retrieval: does a retrieved chunk contain the expected keywords?

| Document | Method | k=1 | k=3 |
| --- | --- | --- | --- |
| Technical README | TF-IDF | 0.75 | 1.00 |
| Technical README | Embedding | 0.50 | 0.58 |
| Technical README | Hybrid | 0.75 | 1.00 |
| Book chapter (narrative) | TF-IDF | 0.60 | 0.60 |
| Book chapter (narrative) | Embedding | 0.80 | 1.00 |
| Book chapter (narrative) | Hybrid | 0.60 | 0.80 |

### Answers: keyword accuracy with `gemini-3.5-flash-lite`

| Document | Method | Keyword accuracy | Correct refusals |
| --- | --- | --- | --- |
| Technical README | TF-IDF | 0.92 | 1.00 |
| Book chapter (narrative) | TF-IDF | 0.40 | 1.00 |
| Book chapter (narrative) | Embedding | 1.00 | 1.00 |

### What the numbers show

Keyword search works best on the technical README, which is full of exact names and numbers such as `MAX_UPLOAD_MB`. On narrative text with reworded questions, embeddings find the right passages far more often, and the answers improve accordingly. With keyword search, the model refused to answer questions whose answer was in the book, because the retrieved chunks did not contain it; it did not make anything up.

### Embedding cache

Chunk and question embeddings are cached on disk in `.cache/embeddings`, keyed by a hash of the texts, the model and the task type. The first run on a document waits for Gemini's free-tier rate limit; later runs reuse the cache. Changing the text, the chunk size or the model produces a new key, so stale embeddings are never used.

### About the book chapter

The book chapter comes from a copyrighted book, so neither the text nor the cache is included in this repository. To reproduce the results, put your own copy of the text in `data/sapiens_part.txt`. Only the question file (`eval/sapiens_questions.json`) is included.

## Known limitations

- The question sets are small (12 answerable README questions, 5 answerable book questions), so every number is indicative only. One question moves a result by 8 to 20 percent.
- The questions were written by the author, and the keyword check measures words, not meaning: a correct answer phrased differently counts as a miss.
- `hit rate` is not reported because it is always 1.0 with a single document.
- A follow-up question costs one extra Gemini request (to rewrite it), and the rewriting is not evaluated yet.
- Indexing with embeddings blocks until it finishes; long documents on the free tier can take minutes.
- The embeddings and the answers need Gemini, whose free tier has request limits.
- Indexed documents live in server memory and are lost on restart.

## Development

Install the pre-commit hook once, so ruff checks and formats the code before every commit:

```
pip install pre-commit ruff
pre-commit install
```

Run the checks by hand:

```
ruff check . --fix
ruff format .
pytest -v
cd frontend
npm run build
```

The tests use fake Gemini responses, so they need no API key. On every push and pull request, GitHub Actions runs ruff, the Python tests and the frontend build. The `main` branch only accepts changes through pull requests.

## Project structure

```
api.py                     FastAPI backend
src/                       loader, chunker, retrievers, embedding cache, answerer, evaluation
frontend/                  React chat page
eval/                      evaluation questions
run_eval.py                answer evaluation
compare_retrievers.py      retrieval comparison (TF-IDF, embeddings, hybrid)
compare_settings.py        chunk size and top-k comparison
tests/                     pytest suite
ruff.toml                  linter and formatter settings
.pre-commit-config.yaml    pre-commit hooks
```
# Ask My Docs

![CI](https://github.com/Jihanvall/ask-my-docs/actions/workflows/ci.yml/badge.svg)

A question-answering system over your own documents. You upload a `.txt`, `.md` or text-based `.pdf` file, ask a question, and get an answer with the sources it came from. If the documents do not contain the answer, it says so instead of guessing.

## How it works

1. **Loading** (`src/loader.py`): reads the file and extracts its text. Scanned PDFs without readable text are rejected with a clear message.
2. **Chunking** (`src/chunker.py`): splits the text into overlapping chunks of words, each keeping its source file and position.
3. **Retrieval**: finds the chunks closest to the question. Three methods are available:
   - TF-IDF (`src/retriever.py`): matches words, runs locally.
   - Embeddings (`src/embedding_retriever.py`): matches meaning using Gemini embeddings.
   - Hybrid (`src/hybrid_retriever.py`): combines both rankings with reciprocal rank fusion.
4. **Answering** (`src/answerer.py`): sends the question and the top chunks to Gemini, which must answer only from them and cite the excerpts. With no relevant chunk, it answers "I don't know" without calling Gemini.
5. **API and interface**: a FastAPI backend (`api.py`) and a React page (`frontend/`).

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

Open `http://localhost:5173`, upload a document and ask a question. The API documentation is at `http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Status and number of indexed documents |
| POST | `/documents` | Uploads a file and indexes it, returns a `session_id` |
| POST | `/ask` | Answers a question for a `session_id`, with sources |

## Evaluation

The evaluation scripts take a document and a question file, so the same pipeline can be measured on different texts:

```
python run_eval.py --doc data/readme.md --questions eval/questions.json
python run_eval.py --method embedding --doc data/sapiens_part.txt --questions eval/sapiens_questions.json
python compare_retrievers.py --doc data/sapiens_part.txt --questions eval/sapiens_questions.json
```

`--method` chooses the retrieval method: `tfidf` (default), `embedding` or `hybrid` .

### Retrieval: does a retrieved chunk contain the expected keywords?


| Document | Method | k=1 | k=3 |
| --- | --- | --- | --- |
| This README (technical) | TF-IDF | 0.75 | 1.00 |
| This README (technical) | Embedding | 0.50 | 0.58 |
| This README (technical) | Hybrid | 0.75 | 1.00 |
| Book chapter (narrative) | TF-IDF | 0.60 | 0.60 |
| Book chapter (narrative) | Embedding | 0.80 | 1.00 |
| Book chapter (narrative) | Hybrid | 0.60 | 0.80 |

### Answers: keyword accuracy with `gemini-3.5-flash-lite`

| Document | Method | Keyword accuracy | Correct refusals |
| --- | --- | --- | --- |
| This README (technical) | TF-IDF | 0.92 | 1.00 |
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
- The embeddings and the answers need Gemini, whose free tier has request limits.
- Indexed documents live in server memory and are lost on restart.


## Tests

```
pytest -v
```

The tests use fake Gemini responses, so they need no API key. GitHub Actions runs them on every push and pull request.

## Project structure

```
api.py                  FastAPI backend
src/                    loader, chunker, retrievers, answerer, evaluation
frontend/               React page
eval/questions.json     evaluation questions
run_eval.py             answer evaluation
compare_retrievers.py   retrieval comparison
tests/                  pytest suite
```
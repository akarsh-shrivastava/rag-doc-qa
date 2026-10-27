# RAG Document Q&A Bot

A web app where you upload PDFs and ask questions about them in plain English. Answers come only from the documents, and every answer shows which page it came from. Built as part of an AI Engineer portfolio.

## Live Demo

[Link to Streamlit App](https://as-rag-doc-app.streamlit.app/)

No PDF handy? Click "Try with sample documents" in the sidebar. The app runs on free tiers, so it can be slow to wake up or hit rate limits when busy.

## Screenshots

| [![Landing](screenshots/landing.png)](screenshots/landing.png) | [![Cross Document Answer](screenshots/cross-doc.png)](screenshots/cross-doc.png) |
|---|---|
| [![Follow-up Question](screenshots/follow-up.png)](screenshots/follow-up.png) | [![Out of Scope Question](screenshots/refusal.png)](screenshots/refusal.png) |

## Features

- **Document Q&A** — upload one or more PDFs and ask questions in natural language
- **Source citations** — every answer lists the file and page it was pulled from, with a text preview
- **Multiple PDFs** — questions are searched across all uploaded documents at once
- **Follow-up questions** — things like "what about weekends?" work after a previous question
- **Stays inside the document** — if the answer isn't in the PDFs, the bot says so instead of guessing
- **Sample documents** — two small PDFs and example questions built in for a quick try

## Tech Stack

| Layer           | Tool                      |
|-----------------|---------------------------|
| Frontend        | Streamlit                 |
| AI Model        | Google Gemini Flash       |
| Embeddings      | Gemini Embedding 001      |
| Orchestration   | LangChain                 |
| Vector store    | FAISS                     |
| PDF parsing     | pypdf                     |
| Deployment      | Streamlit Community Cloud |

## How It Works

1. User uploads one or more PDFs, and text is extracted page by page (page number and filename are kept with each page)
2. The text is split into overlapping chunks (1000 characters, 200 overlap)
3. Each chunk is converted to an embedding with Gemini and stored in a FAISS index
4. A question is embedded the same way, and the closest chunks are fetched from the index
5. The chunks and the question go to Gemini with a prompt telling it to answer only from that context
6. The answer is shown along with the pages the chunks came from

## Chunking

Splitting by character count alone cut rules in half or mixed two sections into one chunk. Adding a section-level separator ahead of the paragraph and sentence separators gave much cleaner chunks on structured documents like the Laws of Cricket, which I used for testing.

Page numbers are kept by splitting page-level documents instead of one big text string. That is what makes citations possible.

## Follow-up Questions

Retrieval happens before the model sees anything, so a vague follow-up like "and for adults?" would pull the wrong chunks if it was searched as it is. Before retrieval, Gemini rewrites the follow-up into a standalone question using the recent chat history, and that rewritten question is used for the search. The answer step never sees the chat history, so the bot still answers only from the documents. The first question skips this step.

## Reliability

- Embeddings are generated in small batches with pauses to stay inside free-tier rate limits
- API errors are turned into short readable messages, and the full error goes to the server log
- Scanned (image-only) PDFs and oversized uploads show a clear error
- The index for the sample documents is built once and cached, so visitors don't use up API quota

## Known Limitations

- Questions that need facts from two different places can miss one of them, because the whole question is embedded as a single vector. Retrieving 6 chunks instead of 3 helps. Splitting the question into sub-queries is the planned fix.
- Scanned PDFs are not supported (no OCR)
- Tables and complex layouts don't extract well as plain text
- Nothing is saved between sessions, so uploaded files are re-indexed each time

## Local Setup

```
git clone <YOUR-REPO-URL>
cd rag-doc-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=<gemini_api_key>
```

Run the app:

```
streamlit run app.py
```

## Project Structure

```
rag-doc-qa/
|_ app.py
|_ rag_pipeline.py
|_ requirements.txt
|_ sample_docs/
|  |_ harborview_membership_guide.pdf
|  |_ harborview_facilities_rules.pdf
|_ screenshots/
|_ LICENSE
|_ README.md
```

## License

MIT

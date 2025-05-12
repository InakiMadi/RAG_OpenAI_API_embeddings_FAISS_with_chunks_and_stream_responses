## RAG OpenAI API embeddings FAISS with chunks and stream response

First of all, install all requirements.

> pip install -r requirements.txt

This is a simple RAG example using OpenAI API, with chunks and possible stream responses, using OpenAI for embeddings and FAISS for vector database. Retrieves information from a CV and answers queries with respect to the CV.

We have an OpenAI client (API key needed for the user, not uploaded for safety reasons), refactored in a clean "private" OpenAIClient class.

### Details of the project

- Data Ingestion: PdfReader.
- Latency: Chunking, response as stream.
- Chat completions: OpenAI gpt-3.5-turbo.
- Embeddings: OpenAI text-embedding-3-small.
- Knowledge basis: FAISS vector database.
- Relevant chunks retrieval: FAISS index built for similarity search.

## RAG OpenAI API embeddings FAISS with chunks and stream response

First of all, install all requirements.

> pip install -r requirements.txt

This is a simple RAG example using OpenAI API, with chunks and possible stream responses, using OpenAI for embeddings and FAISS for simple vector database.

We have:

1. An OpenAI client (API key needed for the user, not uploaded for safety reasons), refactored in a clean private OpenAIClient class.
2. RAG CV FAISS. Retrieves information from a CV and answers queries with respect to the CV.
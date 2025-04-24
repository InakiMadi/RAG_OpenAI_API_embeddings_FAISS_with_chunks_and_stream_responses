from PyPDF2 import PdfReader
from src.openai_client import OpenAIClient
import numpy as np
import faiss
from typing import Optional, List


class RAGCV_FAISS:
    def __init__(self, client: OpenAIClient, cv_path: str):
        self.client_OpenAIClient = client

        # Store in memory for simplicity
        self.text = RAGCV_FAISS.extract_cv(cv_path)
        self.chunks = RAGCV_FAISS.chunk_text(self.text)
        self.embeddings = [self.get_embedding(chunk) for chunk in self.chunks]
        self.index = RAGCV_FAISS.build_index(self.embeddings)

    # Extract text from PDF
    @staticmethod
    def extract_cv(cv_path: str) -> str:
        reader = PdfReader(cv_path)
        return " ".join(page.extract_text() for page in reader.pages)

    # Split text into chunks
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 25) -> List[str]:  # Short chunk_size to simulate bigger texts (CVs are usually short).
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Generate embeddings using OpenAI
    def get_embedding(self, text: str) -> List[float]:
        response = self.client_OpenAIClient.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    # Build FAISS index for similarity search
    @staticmethod
    def build_index(embeddings: List[List[float]]) -> faiss:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        embeddings_array: np.ndarray = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        return index

    def get_relevant_chunks(self, query: str, k: int = 2) -> List[str]:
        query_embedding: List[float] = self.get_embedding(query)
        indices: np.ndarray
        _distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

    # Answer questions using RAG
    def ask_about_cv(self, question: str) -> Optional[str]:
        relevant_chunks = self.get_relevant_chunks(question)

        cv_context = "Answer using ONLY this CV context:\n" + "\n".join(relevant_chunks)
        self.client_OpenAIClient.add_context(cv_context)

        answer: Optional[str] = self.client_OpenAIClient.query(question)
        if answer:
            return answer

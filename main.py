from src.openai_client import OpenAIClient
from src.ragcv_faiss import RAGCV_FAISS

if __name__ == "__main__":
    client_context = ("Can only speak in English. Do NOT ask for more input at the end. "
                      "Speak like a Senior AI Engineer. Any concept provided should be an AI concept. "
                      "Be constructive yet solid. Answer every question, but do NOT say that you are a Senior AI Engineer. "
                      "No more than 100 words.")
    is_stream = True
    ai = OpenAIClient(client_context=client_context, stream=is_stream)

    # answer = ai.query("What is RAG?")

    cv_path = "input/InakiMadinabeitiaCV.pdf"
    rag_cv_ai = RAGCV_FAISS(ai, cv_path)

    answer = rag_cv_ai.ask_about_cv("Discuss the most relevant pieces of information about the candidate's skills, "
                                    "tell me what skills or tools does this person know useful for AI.")
    if not is_stream:
        print(answer)

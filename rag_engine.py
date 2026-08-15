import os
import time
import lancedb
from pathlib import Path
from fastembed import TextEmbedding
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "lancedb_data"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

class RAGResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    confidence_score: float
    retrieval_strategy_used: str
    total_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float

class VoiceRAGSystem:
    def __init__(self):
        # Fallback to C:\Users\Owner\lancedb_data if not found in project folder
        if (DB_PATH / "msmarco_rag.lance").exists() or (DB_PATH).exists():
            target_path = str(DB_PATH)
        else:
            alt_path = Path("C:/Users/Owner/lancedb_data")
            target_path = str(alt_path if alt_path.exists() else DB_PATH)

        self.db = lancedb.connect(target_path)
        self.table = self.db.open_table("msmarco_rag")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env file.")
        self.groq_client = Groq(api_key=api_key)
        self.SIMILARITY_THRESHOLD = 0.45

    def run_query(self, user_question: str) -> RAGResponse:
        t_start = time.perf_counter()

        # Step 1: Dense Query Embedding
        query_vector = list(self.embedding_model.embed([user_question]))[0].tolist()

        # Step 2: Vector Search
        t_ret_start = time.perf_counter()
        results = self.table.search(query_vector).limit(3).to_list()
        t_ret_end = time.perf_counter()
        retrieval_ms = (t_ret_end - t_ret_start) * 1000

        # Step 3: Guardrail Check 1 - Empty Results
        if not results:
            t_end = time.perf_counter()
            return RAGResponse(
                query=user_question,
                answer="I cannot answer as no relevant records were found in the dataset.",
                grounded=False,
                confidence_score=0.0,
                retrieval_strategy_used="none",
                total_latency_ms=round((t_end - t_start) * 1000, 2),
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=0.0
            )

        best_match = results[0]
        similarity_score = 1.0 - best_match.get("_distance", 1.0)

        # Step 4: Guardrail Check 2 - Off-Topic / Low Confidence
        if similarity_score < self.SIMILARITY_THRESHOLD:
            t_end = time.perf_counter()
            return RAGResponse(
                query=user_question,
                answer="Question is ungrounded or out-of-scope for the MSMARCO dataset.",
                grounded=False,
                confidence_score=round(similarity_score, 3),
                retrieval_strategy_used="guardrail_blocked",
                total_latency_ms=round((t_end - t_start) * 1000, 2),
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=0.0
            )

        parent_context = best_match.get("parent_context", best_match["text"])

        # Step 5: Fast LLM Inference (Groq Llama-3.1-8B-Instant)
        t_llm_start = time.perf_counter()
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a grounded RAG engine. Answer the question in exactly 1 concise sentence using ONLY the provided context. If the context does not contain the answer, say 'Information not available.'"
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{parent_context}\n\nQuestion: {user_question}"
                }
            ],
            max_tokens=60,
            temperature=0.0
        )
        t_llm_end = time.perf_counter()
        llm_ms = (t_llm_end - t_llm_start) * 1000
        t_end = time.perf_counter()

        return RAGResponse(
            query=user_question,
            answer=response.choices[0].message.content.strip(),
            grounded=True,
            confidence_score=round(similarity_score, 3),
            retrieval_strategy_used=best_match.get("strategy", "hierarchical_parent_child"),
            total_latency_ms=round((t_end - t_start) * 1000, 2),
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=round(llm_ms, 2)
        )

if __name__ == "__main__":
    rag = VoiceRAGSystem()
    sample = rag.run_query("What is high blood pressure?")
    print(f"\nResult:\nAnswer: {sample.answer}\nGrounded: {sample.grounded}\nTotal Latency: {sample.total_latency_ms} ms")

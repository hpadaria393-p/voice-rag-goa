import os
import numpy as np
from dotenv import load_dotenv
from rag_engine import VoiceRAGSystem

load_dotenv()

print("Initializing Voice RAG Pipeline for Benchmarking...")
rag = VoiceRAGSystem()

test_queries = [
    "What are common causes of high blood pressure?",
    "How does the circulatory system function?",
    "What is the capital of Mars?",                  # Guardrail test
    "Explain the symptoms of vitamin deficiency.",
    "Can you bake a chocolate cake?",                # Guardrail test
    "What are the benefits of cardiovascular exercise?",
    "How do computers process binary instructions?",
    "Tell me the recipe for sushi.",                 # Guardrail test
    "What causes temperature variations in oceans?",
    "How does photosynthesis convert light energy?"
] * 4

latencies = []
retrieval_latencies = []
llm_latencies = []
guardrail_triggers = 0

print(f"\nStarting benchmark across {len(test_queries)} queries...")

for i, q in enumerate(test_queries):
    res = rag.run_query(q)
    latencies.append(res.total_latency_ms)
    retrieval_latencies.append(res.retrieval_latency_ms)
    llm_latencies.append(res.llm_latency_ms)
    
    if not res.grounded:
        guardrail_triggers += 1

    print(f"[{i+1:02d}/{len(test_queries)}] Latency: {res.total_latency_ms:>6.2f}ms | Grounded: {str(res.grounded):<5} | Strategy: {res.retrieval_strategy_used}")

latencies.sort()
p50 = float(np.percentile(latencies, 50))
p70 = float(np.percentile(latencies, 70))
p100 = float(np.max(latencies))

avg_retrieval = float(np.mean(retrieval_latencies))
avg_llm = float(np.mean([l for l in llm_latencies if l > 0])) if any(l > 0 for l in llm_latencies) else 0.0

print("\n=================== LATENCY BENCHMARK REPORT ===================")
print(f"Total Test Runs     : {len(test_queries)}")
print(f"Guardrail Triggers  : {guardrail_triggers} (Off-domain questions safely blocked)")
print(f"Avg Retrieval Time  : {avg_retrieval:.2f} ms")
print(f"Avg LLM Time        : {avg_llm:.2f} ms")
print("----------------------------------------------------------------")
print(f"P50 Latency (Median): {p50:.2f} ms")
print(f"P70 Latency (70th)  : {p70:.2f} ms")
print(f"P100 Latency (Worst): {p100:.2f} ms")
print("================================================================")

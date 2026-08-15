import os
import lancedb
from datasets import load_dataset
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

print("==================================================")
print("  STEP 1: INITIALIZING DATABASE & EMBEDDING MODEL  ")
print("==================================================")
db = lancedb.connect("./lancedb_data")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("\n==================================================")
print("  STEP 2: STREAMING & CHUNKING MSMARCO-XI DATASET  ")
print("==================================================")
dataset_stream = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)

records = []
target_samples = 150
count = 0

for item in dataset_stream:
    if count >= target_samples:
        break

    passage = item.get("passage", "")
    if not passage and "passages" in item:
        passages_dict = item.get("passages", {})
        if isinstance(passages_dict, dict):
            passage = " ".join(passages_dict.get("passage_text", [])) or " ".join(passages_dict.get("English_passages", []))
        elif isinstance(passages_dict, list):
            passage = " ".join(passages_dict)

    if not passage or len(passage.strip()) < 30:
        continue

    count += 1
    
    # 1. Semantic Sentence Chunking
    sentences = [s.strip() for s in passage.split(".") if len(s.strip()) > 15]
    if not sentences:
        sentences = [passage.strip()]

    # 2. Hierarchical Parent-Child Mapping
    parent_text = passage
    for c_idx, sent in enumerate(sentences):
        records.append({
            "id": f"doc_{count}_c{c_idx}",
            "text": sent,
            "parent_context": parent_text,
            "strategy": "hierarchical_parent_child",
            "query_id": str(item.get("query_id", count)),
            "doc_length": len(passage)
        })

print(f"-> Extracted {len(records)} granular chunks across {count} source documents.")

print("\n==================================================")
print("  STEP 3: COMPUTING DENSE VECTOR EMBEDDINGS        ")
print("==================================================")
texts = [r["text"] for r in records]
embeddings = list(embedding_model.embed(texts))

for r, emb in zip(records, embeddings):
    r["vector"] = emb.tolist()

print("\n==================================================")
print("  STEP 4: SAVING TABLE 'msmarco_rag' TO LANCEDB    ")
print("==================================================")
table = db.create_table("msmarco_rag", data=records, mode="overwrite")
print(f"✅ Success! Table 'msmarco_rag' created with {len(records)} vectors.")

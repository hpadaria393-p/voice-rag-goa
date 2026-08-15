import os
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from rag_engine import VoiceRAGSystem
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Voice-Enabled RAG Harness")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = VoiceRAGSystem()

def transcribe_elevenlabs(audio_bytes: bytes) -> str:
    """Fallback / Primary ElevenLabs Speech-to-Text handler"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return ""
    
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("text", "")
    except Exception:
        pass
    return ""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/query-text")
async def query_text(data: dict):
    question = data.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)
    
    res = rag_system.run_query(question)
    return res.dict()

@app.post("/api/query-voice")
async def query_voice(audio_file: UploadFile = File(...)):
    t0 = time.perf_counter()
    audio_bytes = await audio_file.read()
    
    # 1. Transcribe speech to text
    transcription = transcribe_elevenlabs(audio_bytes)
    t_stt = (time.perf_counter() - t0) * 1000

    if not transcription:
        return {
            "query": "[Voice transcription failed or missing key]",
            "answer": "Could not parse audio input. Please verify ElevenLabs API key.",
            "grounded": False,
            "confidence_score": 0.0,
            "retrieval_strategy_used": "none",
            "total_latency_ms": round(t_stt, 2),
            "retrieval_latency_ms": 0.0,
            "llm_latency_ms": 0.0
        }
    
    # 2. Run through RAG Harness
    rag_res = rag_system.run_query(transcription)
    total_latency = (time.perf_counter() - t0) * 1000
    
    result = rag_res.dict()
    result["total_latency_ms"] = round(total_latency, 2)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

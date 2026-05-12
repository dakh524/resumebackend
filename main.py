from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import ats
import rag
import os
import requests as req

app = FastAPI(title="ResumeAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if rag.collection.count() == 0:
        print("ChromaDB empty. Loading data...")
        rag.load_data_to_chroma()
        print("Done loading!")
    else:
        print(f"ChromaDB ready: {rag.collection.count()} questions")

@app.get("/")
async def root():
    return {"message": "ResumeAI Backend is Running"}

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        content = await file.read()
        analysis = ats.analyze_resume(content)
        role = analysis["role"]
        questions = rag.get_questions(role, top_k=10)
        tips = rag.get_linkedin_tips(role)
        return {
            "ats_score": analysis["ats_score"],
            "role": role,
            "matched_keywords": analysis["matched_keywords"],
            "missing_keywords": analysis["missing_keywords"],
            "interview_questions": questions,
            "linkedin_tips": tips
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    interview_question: str = ""
    doubt: str = ""

@app.post("/chat")
async def chat(request: ChatRequest):
    doubt = request.doubt
    question = request.interview_question

    # Step 1: RAG
    try:
        from rag import collection, model
        results = collection.query(
            query_texts=[f"{question} {doubt}"],
            n_results=3
        )
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            if docs and len(docs[0]) > 50:
                return {
                    "answer": docs[0],
                    "source": "rag"
                }
    except Exception as e:
        print(f"RAG error: {e}")

    # Step 2: Gemini
    try:
        import google.generativeai as genai
        genai.configure(
            api_key=os.environ.get("GEMINI_API_KEY","")
        )
        m = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are a career and interview expert.
Answer only resume and interview questions.
Context: {question}
Doubt: {doubt}
Answer in 3-5 clear lines."""
        response = m.generate_content(prompt)
        return {
            "answer": response.text,
            "source": "gemini"
        }
    except Exception as e:
        print(f"Gemini error: {e}")

    # Step 3: OpenRouter
    try:
        r = req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY','')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{
                    "role": "user",
                    "content": f"Interview doubt: {doubt}. Context: {question}. Answer in 3-5 lines."
                }]
            },
            timeout=10
        )
        answer = r.json()['choices'][0]['message']['content']
        return {"answer": answer, "source": "openrouter"}
    except Exception as e:
        print(f"OpenRouter error: {e}")

    # Step 4: Fallback
    return {
        "answer": "I couldn't find a specific answer right now. Please check GeeksforGeeks or YouTube for this topic 📚",
        "source": "fallback"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

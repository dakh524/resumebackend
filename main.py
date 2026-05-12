from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ats
import rag
import os

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

from pydantic import BaseModel

import chat

@app.post("/chat")
async def chat(request: dict):
    interview_question = request.get("interview_question", "")
    doubt = request.get("doubt", "")
    
    # Step 1: Search RAG (ChromaDB)
    try:
        from rag import collection
        results = collection.query(
            query_texts=[f"{interview_question} {doubt}"],
            n_results=3
        )
        if results['documents'] and results['documents'][0]:
            answer = results['documents'][0][0]
            if len(answer) > 50:
                return {"answer": answer, "source": "rag"}
    except Exception as e:
        print(f"RAG Error: {e}")
    
    # Step 2: Gemini API
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are a career and interview expert.
Only answer resume and interview related questions.
Interview question context: {interview_question}
Student doubt: {doubt}
Give a clear helpful answer in 3-5 lines."""
        response = model.generate_content(prompt)
        if response.text:
            return {"answer": response.text, "source": "gemini"}
    except Exception as e:
        print(f"Gemini Error: {e}")
    
    # Step 3: OpenRouter
    try:
        import requests as req
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", 
            "content": f"Answer this interview doubt: {doubt} \nContext: {interview_question}"}]
        }
        r = req.post("https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=10)
        answer = r.json()['choices'][0]['message']['content']
        return {"answer": answer, "source": "openrouter"}
    except Exception as e:
        print(f"OpenRouter Error: {e}")
    
    # Step 4: Fallback
    return {
        "answer": "Sorry, I couldn't find a specific answer. Please check GeeksforGeeks or YouTube for this topic 📚",
        "source": "fallback"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

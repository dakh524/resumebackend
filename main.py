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
        print("Data loaded successfully!")
    else:
        print(f"ChromaDB ready with {rag.collection.count()} questions.")

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

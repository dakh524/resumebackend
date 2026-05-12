from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ats
import rag
import os

app = FastAPI(title="ResumeAI API")

# Enable CORS for mobile app (Allow all origins as requested)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ResumeAI Backend is Running"}

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content
        content = await file.read()
        
        # 1. ATS Analysis (Role, Score, Keywords)
        analysis = ats.analyze_resume(content)
        role = analysis["role"]
        
        # 2. RAG Logic (Questions & Tips)
        questions = rag.get_questions(role, top_k=5)
        tips = rag.get_linkedin_tips(role)
        
        # Combine results
        result = {
            "ats_score": analysis["ats_score"],
            "role": role,
            "matched_keywords": analysis["matched_keywords"],
            "missing_keywords": analysis["missing_keywords"],
            "interview_questions": questions,
            "linkedin_tips": tips
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail="Error processing resume analysis")

if __name__ == "__main__":
    # Check if data exists, if not, suggest running scraper
    if not os.path.exists("scraped_data.json"):
        print("Warning: scraped_data.json not found. Results will be limited.")
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

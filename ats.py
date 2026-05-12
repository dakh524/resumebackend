import pdfplumber
import io
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined keyword lists for roles to help with extraction
KEYWORDS_DB = {
    "Python Developer": ["Python", "Django", "Flask", "Pandas", "NumPy", "SQL", "Git", "FastAPI"],
    "Java Developer": ["Java", "Spring Boot", "Hibernate", "Microservices", "Maven", "Jenkins"],
    "React Developer": ["React", "JavaScript", "TypeScript", "Redux", "Tailwind", "HTML", "CSS"],
    "Data Scientist": ["Python", "R", "Machine Learning", "Statistics", "Scikit-learn", "TensorFlow"],
    "ML Engineer": ["PyTorch", "TensorFlow", "Keras", "Model Deployment", "MLOps", "Computer Vision"],
    "Software Engineer": ["Algorithms", "Data Structures", "System Design", "OOP", "Testing", "Agile"]
}

def extract_text_from_pdf(file_content):
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def detect_role(text):
    text = text.lower()
    scores = {}
    for role, keywords in KEYWORDS_DB.items():
        score = sum(1 for k in keywords if k.lower() in text)
        scores[role] = score
    
    # Return highest scoring role, default to Software Engineer
    best_role = max(scores, key=scores.get) if any(scores.values()) else "Software Engineer"
    return best_role

def get_ats_score(resume_text, role):
    # Get keywords for the detected role
    target_keywords = KEYWORDS_DB.get(role, KEYWORDS_DB["Software Engineer"])
    
    # Check matched vs missing
    matched = [k for k in target_keywords if k.lower() in resume_text.lower()]
    missing = [k for k in target_keywords if k.lower() not in resume_text.lower()]
    
    # AI Similarity Score (Mock JD based on role)
    jd_mock = f"Experienced {role} with skills in {', '.join(target_keywords)}. Must have strong problem solving abilities."
    
    resume_emb = model.encode(resume_text)
    jd_emb = model.encode(jd_mock)
    
    similarity = util.cos_sim(resume_emb, jd_emb).item()
    # Normalize to 0-100 and add a little weight for matched keywords
    score = int(similarity * 80 + (len(matched) / len(target_keywords)) * 20)
    
    return min(score, 100), matched, missing

def analyze_resume(file_content):
    text = extract_text_from_pdf(file_content)
    role = detect_role(text)
    score, matched, missing = get_ats_score(text, role)
    
    return {
        "ats_score": score,
        "role": role,
        "matched_keywords": matched,
        "missing_keywords": missing
    }

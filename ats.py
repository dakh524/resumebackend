import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KEYWORDS_DB = {
    "Python Developer": ["Python", "Django", "Flask", "Pandas", "NumPy", "SQL", "Git", "FastAPI"],
    "Java Developer": ["Java", "Spring Boot", "Hibernate", "Microservices", "Maven"],
    "React Developer": ["React", "JavaScript", "TypeScript", "Redux", "HTML", "CSS"],
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Scikit-learn", "TensorFlow"],
    "ML Engineer": ["PyTorch", "TensorFlow", "Keras", "MLOps", "Computer Vision"],
    "Software Engineer": ["Algorithms", "Data Structures", "System Design", "OOP", "Testing", "Agile"],
    "Electronics Engineer": ["Circuit Design", "PCB", "VHDL", "Microcontrollers", "Embedded C", "MATLAB"],
    "Mechanical Engineer": ["CAD", "AutoCAD", "SolidWorks", "Thermodynamics", "FEA"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Linux"],
    "Backend Developer": ["Node.js", "Express", "PostgreSQL", "MongoDB", "Docker", "API"],
    "Marketing Manager": ["SEO", "SEM", "Google Analytics", "Social Media"],
    "Business Analyst": ["SQL", "Tableau", "Excel", "Agile", "Requirements"]
}

def extract_text_from_pdf(file_content):
    try:
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()
    except Exception as e:
        print(f"Extraction error: {e}")
        return ""

def detect_role(text):
    text_lower = text.lower()
    scores = {}
    for role, keywords in KEYWORDS_DB.items():
        score = sum(1 for k in keywords if k.lower() in text_lower)
        scores[role] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Software Engineer"

def get_ats_score(resume_text, role):
    keywords = KEYWORDS_DB.get(role, KEYWORDS_DB["Software Engineer"])
    matched = [k for k in keywords if k.lower() in resume_text.lower()]
    missing = [k for k in keywords if k.lower() not in resume_text.lower()]
    jd = f"Experienced {role} with skills in {', '.join(keywords)}"
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_text[:1000], jd])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        similarity = 0.5
    score = int(similarity * 80 + (len(matched) / max(len(keywords), 1)) * 20)
    return min(score, 100), matched, missing

def analyze_resume(file_content):
    text = extract_text_from_pdf(file_content)
    if not text:
        return {"ats_score": 0, "role": "Unknown", "matched_keywords": [], "missing_keywords": []}
    role = detect_role(text)
    score, matched, missing = get_ats_score(text, role)
    return {"ats_score": score, "role": role, "matched_keywords": matched, "missing_keywords": missing}

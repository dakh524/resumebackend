import pdfplumber
import io
import re

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

def calculate_ats_breakdown(text, role):
    text_lower = text.lower()
    
    # 1. Keyword Match Score (40 pts)
    keywords = KEYWORDS_DB.get(role, KEYWORDS_DB["Software Engineer"])
    matched = [k for k in keywords if k.lower() in text_lower]
    missing = [k for k in keywords if k.lower() not in text_lower]
    keyword_score = (len(matched) / max(len(keywords), 1)) * 40
    
    # 2. Format & Parseability Score (25 pts)
    format_score = 25
    # Basic detection for complex layouts
    if "\t" in text or "  " in text: format_score -= 5 # Columns/Tables hint
    if "image" in text_lower or "graphic" in text_lower: format_score -= 5
    # Check for text density (extremely low density might mean images/scans)
    if len(text) < 200: format_score -= 10
    format_score = max(format_score, 0)
    
    # 3. Section Structure Score (20 pts)
    sections = ["Experience", "Education", "Skills", "Summary", "Projects"]
    sections_found = 0
    for s in sections:
        if s.lower() in text_lower:
            sections_found += 4
    section_score = sections_found
    
    # 4. Contact Info Score (15 pts)
    contact_score = 0
    # Email regex
    if re.search(r'[\w\.-]+@[\w\.-]+', text): contact_score += 5
    # Phone regex (basic)
    if re.search(r'\+?\d{10,12}', text): contact_score += 5
    # URL regex (LinkedIn/Portfolio)
    if "linkedin.com" in text_lower or "github.com" in text_lower or "http" in text_lower:
        contact_score += 5
        
    total_score = int(keyword_score + format_score + section_score + contact_score)
    
    # Rating
    rating = "Poor"
    if total_score >= 80: rating = "Excellent"
    elif total_score >= 60: rating = "Good"
    elif total_score >= 40: rating = "Average"
    
    return {
        "ats_score": min(total_score, 100),
        "rating": rating,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "breakdown": {
            "keyword": int(keyword_score),
            "format": int(format_score),
            "section": int(section_score),
            "contact": int(contact_score)
        }
    }

def analyze_resume(file_content):
    text = extract_text_from_pdf(file_content)
    if not text:
        return {
            "ats_score": 0,
            "role": "Unknown",
            "matched_keywords": [],
            "missing_keywords": [],
            "rating": "Poor"
        }
    role = detect_role(text)
    analysis = calculate_ats_breakdown(text, role)
    analysis["role"] = role
    return analysis

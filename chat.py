import os
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
from rag import collection, model

load_dotenv()

# Topic filter keywords
TOPIC_KEYWORDS = [
    "resume", "interview", "job", "career", "skill", "python",
    "java", "react", "javascript", "sql", "api", "algorithm",
    "data structure", "machine learning", "ai", "cloud",
    "aws", "docker", "kubernetes", "git", "agile", "scrum",
    "salary", "hr", "linkedin", "portfolio", "experience",
    "internship", "fresher", "engineer", "developer",
    "manager", "mba", "marketing", "placement", "technical",
    "hello", "hi", "hey", "thanks", "thank you", "help"
]

def is_topic_related(text):
    text = text.lower()
    return any(keyword in text for keyword in TOPIC_KEYWORDS)

def get_ai_answer(question, doubt):
    # STEP 1: Topic Filter
    if not is_topic_related(doubt) and not is_topic_related(question):
        return {
            "answer": "I can only help with resume and interview related questions 😊 Please ask about jobs, skills, or career topics.",
            "source": "fallback"
        }

    # STEP 2: RAG Search (Priority 1)
    try:
        results = collection.query(
            query_texts=[f"{question} {doubt}"],
            n_results=1,
            include=['documents', 'distances']
        )
        
        # Check similarity score (ChromaDB returns distances, smaller is better)
        # Distance < 0.3 means high similarity (~ >0.7 score)
        if results['documents'] and results['distances'] and results['distances'][0][0] < 0.3:
            return {
                "answer": f"Based on our database: {results['documents'][0][0]}",
                "source": "rag"
            }
    except Exception as e:
        print(f"RAG error: {e}")

    # STEP 3: Gemini API (Priority 2)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            You are a career and interview expert. 
            Answer only resume and interview related questions.
            Context question: {question}
            Student doubt: {doubt}
            Give a clear, helpful explanation in 3-5 lines.
            """
            response = gemini_model.generate_content(prompt)
            if response.text:
                return {
                    "answer": response.text,
                    "source": "gemini"
                }
        except Exception as e:
            print(f"Gemini error: {e}")

    # STEP 4: OpenRouter API (Priority 3)
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [
                        {"role": "user", "content": f"Context: {question}\nDoubt: {doubt}\nAnswer in 3-5 lines as a career expert."}
                    ]
                },
                timeout=10
            )
            data = response.json()
            if 'choices' in data:
                return {
                    "answer": data['choices'][0]['message']['content'],
                    "source": "openrouter"
                }
            else:
                print(f"OpenRouter API error: {data}")
        except Exception as e:
            print(f"OpenRouter exception: {e}")

    # STEP 5: Fallback
    return {
        "answer": "Sorry, I couldn't find a specific answer. Please check GeeksforGeeks, YouTube, or Interviewbit for this topic 📚",
        "source": "fallback"
    }

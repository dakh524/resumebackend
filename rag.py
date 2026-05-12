import chromadb
import json
import os
from shared import model

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="resume_questions")

def load_data_to_chroma():
    if not os.path.exists("scraped_data.json"):
        print("scraped_data.json not found. Run scraper.py first.")
        return

    with open("scraped_data.json", "r") as f:
        data = json.load(f)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    count = 0
    for entry in data:
        role = entry["role"]
        for q in entry["questions"]:
            ids.append(f"q_{count}")
            documents.append(q)
            metadatas.append({"role": role})
            count += 1

    # Embed in batches
    embeddings = model.encode(documents, show_progress_bar=True)
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    print(f"Loaded {count} questions into ChromaDB.")

def get_questions(role, top_k=10):
    # Query ChromaDB based on role and text similarity
    query_text = f"Technical interview questions for a {role} professional"
    query_embedding = model.encode([query_text]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0]
    
    # Fallback questions if database query fails
    return [
        "Can you walk me through your most challenging project?",
        "How do you handle technical debt in your workflow?",
        "What is your approach to learning new technologies quickly?",
        "Describe a time you had to resolve a difficult bug.",
        "How do you ensure code quality in a team environment?"
    ]

def get_linkedin_tips(role):
    # Static tips based on role categories
    tips = {
        "Software": [
            "Highlight your GitHub repository and open source contributions.",
            "Use keywords like React, Python, or Cloud Architecture in your headline.",
            "Add a professional headshot with a neutral background."
        ],
        "Engineer": [
            "List specific CAD tools or simulation software you've mastered.",
            "Include certifications like Six Sigma or PMP.",
            "Detail your hands-on project experience in the summary."
        ],
        "Business": [
            "Quantify your achievements (e.g., 'Increased sales by 20%').",
            "Get endorsements for soft skills like Leadership and Negotiation.",
            "Write a compelling 'About' section that tells your professional story."
        ]
    }
    
    if any(keyword in role for keyword in ["Developer", "React", "Data", "ML", "Python"]):
        return tips["Software"]
    elif "Engineer" in role:
        return tips["Engineer"]
    else:
        return tips["Business"]

def get_answer(question, doubt):
    # Find the most relevant context from our database
    query_text = f"{question} {doubt}"
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    
    if not results['documents'] or len(results['documents'][0]) == 0:
        return "I'm sorry, I don't have specific data on that topic yet. However, generally speaking, this relates to core engineering principles. Could you clarify which part is confusing?"

    matched_q = results['documents'][0][0]
    
    # Simulate a high-quality explanation based on the matched topic
    explanations = {
        "Python": "Python is a high-level interpreted language. The GIL (Global Interpreter Lock) ensures only one thread executes at a time, which simplifies memory management but can limit multi-core performance.",
        "Java": "Java is a class-based, object-oriented language. It runs on the JVM, providing platform independence ('Write Once, Run Anywhere').",
        "React": "React is a JavaScript library for building user interfaces. It uses a Virtual DOM to optimize rendering and state management.",
        "Data Science": "Data Science involves extracting insights from data using statistics, machine learning, and visualization techniques.",
        "Electronics": "This involves the study of electron flow in circuits, utilizing components like transistors, diodes, and microcontrollers to process signals."
    }

    # Find a keyword match for the explanation
    for key, text in explanations.items():
        if key.lower() in matched_q.lower() or key.lower() in doubt.lower():
            return f"Regarding '{matched_q}': {text} For your specific doubt '{doubt}', I recommend focusing on the fundamental implementation details."

    return f"Great question! '{matched_q}' is a common interview topic. To address your doubt about '{doubt}', you should focus on how this concept solves specific technical challenges in real-world projects."

if __name__ == "__main__":
    # If run directly, initialize the DB
    load_data_to_chroma()

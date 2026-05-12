import chromadb
import json
import os
from sentence_transformers import SentenceTransformer

# Initialize Model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    results = collection.query(
        query_texts=[f"interview questions for {role}"],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []

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

if __name__ == "__main__":
    # If run directly, initialize the DB
    load_data_to_chroma()

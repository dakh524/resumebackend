from sentence_transformers import SentenceTransformer
import os

# Load model once for the whole application
# Using a faster/smaller model for low-latency
print("Initializing AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("AI Model Ready.")

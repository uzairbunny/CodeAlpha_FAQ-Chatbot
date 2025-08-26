from flask import Flask, jsonify, render_template, request
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "extracted_faqs"

def load_faqs(data_dir):
    faqs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), 'r', encoding="utf-8") as f:
                try:
                    # Load the JSON data from the file
                    data = json.load(f)
                    # Handle both "question"/"answer" and "Question"/"Answer" formats
                    for item in data:
                        if "question" in item and "answer" in item:
                            faqs.append({"question": item["question"], "answer": item["answer"]})
                        elif "Question" in item and "Answer" in item:
                            faqs.append({"question": item["Question"], "answer": item["Answer"]})
                except json.JSONDecodeError:
                    # If JSON parsing fails, skip this file
                    print(f"Error parsing JSON in file: {file}")
                    continue
    return faqs

# Load FAQs and initialize model outside of any function
faqs = load_faqs(DATA_DIR)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract questions and answers
questions = [f["question"] for f in faqs]
answers = [f["answer"] for f in faqs]

# Generate embeddings for questions
embeddings = model.encode(questions)

app = Flask(__name__)

def chatbot_response(user_input):
    # Generate embedding for user input
    user_embed = model.encode([user_input])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(user_embed, embeddings)[0]
    
    # Find the best match
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    # Return response based on similarity threshold
    if best_score < 0.5:
        return "Sorry, I don't understand your question."
    return answers[best_idx]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=["POST"])
def get_bot_response():
    user_text = request.json.get("message")
    response = chatbot_response(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
from datetime import datetime
import shutil
import re
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key in production

# --- Model Loading ---
@staticmethod
def load_abstractive_model(model_zip="best_bartlarge_model.zip", tokenizer_zip="best_bartlarge_tok.zip"):
    model_path = "best_bartlarge_model"
    tokenizer_path = "best_bartlarge_tok"
    
    if not os.path.exists(model_path):
        try:
            shutil.unpack_archive(model_zip, model_path)
        except Exception as e:
            print(f"Error unzipping model: {e}")
            return None, None
    if not os.path.exists(tokenizer_path):
        try:
            shutil.unpack_archive(tokenizer_zip, tokenizer_path)
        except Exception as e:
            print(f"Error unzipping tokenizer: {e}")
            return None, None
    
    try:
        tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return None, None
    
    return model, tokenizer

abs_model, abs_tokenizer = load_abstractive_model()
if abs_model is None or abs_tokenizer is None:
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abs_model.to(device)

kw_model = KeyBERT()
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- User Management Functions ---
def load_users():
    users_file = "users.json"
    try:
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading users: {e}")
        return {}

def save_users(users):
    try:
        with open("users.json", "w") as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

def register_user(email, password, user_type, passkey=None):
    users = load_users()
    admin_passkey = "secretkey123"
    
    if email in users:
        return False, "User already exists."
    
    if user_type == "Admin" and passkey != admin_passkey:
        return False, "Invalid admin passkey."
    
    stored_type = "admin" if user_type == "Admin" else "user"
    users[email] = {"password": password, "type": stored_type}
    save_users(users)
    return True, f"Registered successfully as {user_type}!"

def check_credentials(email, password, passkey=None):
    users = load_users()
    admin_passkey = "secretkey123"
    
    if email in users:
        stored_password = users[email]["password"]
        user_type = users[email]["type"]
        if stored_password == password:
            if user_type == "admin" and passkey == admin_passkey:
                return "admin"
            elif user_type == "user" and passkey is None:
                return "user"
    return None

# --- Summarization Functions ---
def abstractive_summarize(text, num_keywords=3, format_choice="paragraph"):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    inputs = abs_tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
    summary_ids = abs_model.generate(
        inputs["input_ids"],
        max_length=160,
        min_length=90,
        length_penalty=1.2,
        num_beams=8,
        early_stopping=True,
        repetition_penalty=1.8,
        num_beam_groups=4,
        diversity_penalty=0.8,
        do_sample=False,
        no_repeat_ngram_size=3
    )
    summary = abs_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    summary_sentences = re.split(r'(?<=[.!?])\s+', summary)
    if format_choice == "bullet":
        formatted_summary = "\n".join([f"- {s.strip()}" for s in summary_sentences if s.strip()])
    else:
        formatted_summary = summary
    
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        top_n=min(num_keywords, 5),
        use_mmr=True,
        diversity=0.7
    )
    keywords = [kw[0] for kw in keywords]
    
    return formatted_summary, keywords

def extractive_summarize(text, num_sentences=3, num_keywords=3, format_choice="paragraph"):
    text = str(text).strip()
    sentences = sent_tokenize(text)
    
    if len(sentences) < 1:
        return "No valid sentences found.", [], 0.0
    if len(sentences) < num_sentences:
        num_sentences = len(sentences)
    
    # Sentence vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        sentence_vectors = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        # Handle case where TF-IDF fails (e.g., all sentences are stop words)
        return "Unable to vectorize sentences.", [], 0.0
    
    # Optional: Use sentence embeddings instead (uncomment to use)
    # sentence_vectors = sent_model.encode(sentences)
    
    # Compute centroid (mean vector of all sentences)
    centroid = np.mean(sentence_vectors, axis=0).reshape(1, -1)
    
    # Apply k-NN to find sentences closest to centroid
    k = min(len(sentences), max(1, num_sentences))
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(sentence_vectors)
    distances, indices = knn.kneighbors(centroid)
    
    # Select top sentences based on k-NN ranking
    summary_sentences = [sentences[i] for i in indices[0][:num_sentences]]
    
    # Format the summary
    if format_choice == "bullet":
        formatted_summary = "\n".join([f"- {s.strip()}" for s in summary_sentences])
    else:
        formatted_summary = " ".join(summary_sentences)
    
    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        top_n=min(num_keywords, 5),
        use_mmr=True,
        diversity=0.7
    )
    keywords = [kw[0] for kw in keywords]
    
    # Compute cosine similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text, formatted_summary])
        cosine_similarity_score = np.dot(tfidf_matrix[0], tfidf_matrix[1].T).toarray()[0][0] / (
            np.linalg.norm(tfidf_matrix[0].toarray()) * np.linalg.norm(tfidf_matrix[1].toarray()) + 1e-10
        )
    except ValueError:
        cosine_similarity_score = 0.0
    
    return formatted_summary, keywords, cosine_similarity_score

# --- News Fetching ---
def fetch_cnn_news():
    api_key = "f6f369e1e3f447d59def55272b256e77"
    url = f"https://newsapi.org/v2/everything?domains=cnn.com&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        print(f"Error fetching CNN news: {e}")
        return []

# --- History Functions ---
def save_history(entry):
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []
    entry["user_email"] = session.get("email")
    history.append(entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

def load_history():
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

# --- Feedback Functions ---
def save_feedback(entry):
    feedback_file = "feedback.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            feedback = json.load(f)
    else:
        feedback = []
    entry["user_email"] = session.get("email")
    feedback.append(entry)
    with open(feedback_file, "w") as f:
        json.dump(feedback, f, indent=4)

def load_feedback():
    feedback_file = "feedback.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            return json.load(f)
    return []

# --- Routes ---
@app.route('/')
def home():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    articles = fetch_cnn_news()
    return render_template('index.html', page='home', articles=articles, user_type=session.get("user_type"), email=session.get("email"))

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    if request.method == 'POST':
        print("Form data:", request.form)  # Debug: Check received data
        text = request.form['text']
        summ_type = request.form.get('summ_type', 'abstractive')  # Default to abstractive
        num_keywords = int(request.form.get('num_keywords', 3))  # Default to 3
        format_choice = request.form.get('format_choice', 'paragraph')  # Default to paragraph
        if summ_type == 'extractive':
            num_sentences = int(request.form.get('num_sentences', 3))  # Default to 3
            summary, keywords, cosine_similarity = extractive_summarize(text, num_sentences, num_keywords, format_choice)
            save_history({
                "type": "extractive",
                "input": text[:100],
                "summary": summary,
                "keywords": keywords,
                "cosine_similarity": cosine_similarity,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return render_template('index.html', page='summarize', summary=summary, keywords=keywords, cosine_similarity=cosine_similarity, user_type=session.get("user_type"), email=session.get("email"))
        else:  # abstractive
            summary, keywords = abstractive_summarize(text, num_keywords, format_choice)
            save_history({
                "type": "abstractive",
                "input": text[:100],
                "summary": summary,
                "keywords": keywords,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return render_template('index.html', page='summarize', summary=summary, keywords=keywords, user_type=session.get("user_type"), email=session.get("email"))
    return render_template('index.html', page='summarize', user_type=session.get("user_type"), email=session.get("email"))

@app.route('/history')
def history():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    history = load_history()
    user_history = [h for h in history if h["user_email"] == session.get("email")] if session.get("user_type") != "admin" else history
    return render_template('index.html', page='history', history=user_history, user_type=session.get("user_type"), email=session.get("email"))

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if not session.get("logged_in"):
        return redirect(url_for('login'))
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        if feedback_text.strip():
            save_feedback({
                "feedback": feedback_text,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return render_template('index.html', page='feedback', message="Thank you for your feedback!", user_type=session.get("user_type"), email=session.get("email"))
    feedback_entries = load_feedback()
    user_feedback = [f for f in feedback_entries if f["user_email"] == session.get("email")] if session.get("user_type") != "admin" else feedback_entries
    return render_template('index.html', page='feedback', feedback_entries=user_feedback, user_type=session.get("user_type"), email=session.get("email"))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print(request.form)  # Debug: Print all form data
        email = request.form['email']
        password = request.form['password']
        login_type = request.form['login_type']  # Changed from radio to direct key
        passkey = request.form.get('passkey')
        user_type = "admin" if login_type == "admin" else "user"
        if check_credentials(email, password, passkey if login_type == "admin" else None):
            session["logged_in"] = True
            session["user_type"] = user_type
            session["email"] = email
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        passkey = request.form.get('passkey')
        success, message = register_user(email, password, user_type, passkey)
        if success:
            return render_template('login.html', message=message)
        return render_template('register.html', error=message)
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop("logged_in", None)
    session.pop("user_type", None)
    session.pop("email", None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
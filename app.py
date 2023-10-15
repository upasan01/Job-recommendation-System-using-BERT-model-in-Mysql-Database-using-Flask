from flask import Flask, request, render_template, redirect, url_for
import mysql.connector
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__, static_url_path='/static')

# Database configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="recommendation"
)

cursor = db.cursor()

# Create a table to store job profiles if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_profiles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        profile_name VARCHAR(255),
        lawyer_type VARCHAR(255),            
        description TEXT
    )
""")
db.commit()

# Load the job profiles dataset from the database
cursor.execute("SELECT * FROM job_profiles")
job_profiles_data = cursor.fetchall()
job_profiles = pd.DataFrame(job_profiles_data, columns=['id', 'title', 'lawyer_type', 'description'])

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings for text
def get_bert_embeddings(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to get profile recommendations based on BERT embeddings
def get_profile_recommendations(input_description):
    input_description = input_description.lower()
    if job_profiles.empty:
        return []  # Return an empty list if there are no profiles in the database
    input_embedding = get_bert_embeddings(input_description)
    job_embeddings = [get_bert_embeddings(desc) for desc in job_profiles['lawyer_type']]
    
    # Calculate cosine similarity between input and job profiles
    similarities = [torch.cosine_similarity(input_embedding, job_embedding) for job_embedding in job_embeddings]
    
    # Sort by similarity scores in descending order
    profile_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    
    return job_profiles['title'].iloc[profile_indices[:5]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_profile():
    if request.method == 'POST':
        profile_name = request.form['profile_name']
        lawyer_type = request.form['lawyer_type']
        description = request.form['description']
        cursor.execute("INSERT INTO job_profiles (profile_name,lawyer_type, description) VALUES (%s,%s, %s)", (profile_name, lawyer_type, description))
        db.commit()
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend_profiles():
    if request.method == 'POST':
        profile_description = request.form['profile_description']
        recommended_profiles = get_profile_recommendations(profile_description)
        return render_template('recommendations.html', profile_description=profile_description, recommended_profiles=recommended_profiles)

if __name__ == '__main__':
    app.run(debug=True)

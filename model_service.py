That's an excellent and specific project idea! It's a complex system that combines Machine Learning (BERT) and web development (Flask + MySQL).

Since you're asking me to "make the model using AI", I'll provide you with the core Python code components for the Flask application and the BERT-based recommendation logic. You'll need to set up your files, install libraries, and configure your MySQL database separately.

I'll break down the code into three main files:

app.py: The Flask application logic and API endpoint.

model_service.py: The logic for generating BERT embeddings and calculating similarity.

database_setup.sql: The SQL to set up your MySQL database table.

1. Prerequisites and Setup
Before running the code, you'll need to install the necessary Python libraries:

Bash

pip install Flask transformers sentence-transformers torch scikit-learn numpy mysql-connector-python
The Database Schema
First, you need a MySQL table to store the job listings and their pre-calculated BERT embeddings.

database_setup.sql
(Run this SQL script in your MySQL client)

SQL

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS job_recommender;
USE job_recommender;

-- Create the table for job listings
CREATE TABLE job_listings (
    job_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    -- Embedding column: BERT-Base models often produce 768-dimensional vectors.
    -- Store them as a string (JSON or comma-separated) or a specialized BLOB/VECTOR type if supported.
    -- For simplicity, we'll assume a BLOB in this example, but a TEXT/JSON column is often easier.
    -- In a production environment, use a vector database (like Pinecone, Qdrant) for fast search.
    embedding BLOB NOT NULL
);

-- Example Insertion (You will replace '...' with the actual BERT vector bytes)
-- INSERT INTO job_listings (title, description, embedding) VALUES
-- ('Data Scientist', 'Analyze data and build models...', <BERT_VECTOR_FOR_DS>),
-- ('Python Developer', 'Develop Flask and Django apps...', <BERT_VECTOR_FOR_DEV>);
2. BERT Model Service (model_service.py)
This file handles loading the BERT model (specifically, Sentence-BERT for better sentence similarity) and generating the embeddings.

model_service.py

Python

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained Sentence-BERT model. 
# 'all-MiniLM-L6-v2' is fast and good for quick demos.
MODEL_NAME = 'all-MiniLM-L6-v2' 
model = SentenceTransformer(MODEL_NAME)

def get_embedding(text_list):
    """
    Converts a list of texts (job descriptions or resume text) into numerical embeddings.
    
    Args:
        text_list (list): A list of strings to be embedded.

    Returns:
        numpy.ndarray: A 2D array of embeddings.
    """
    # encode() handles tokenization and model inference
    embeddings = model.encode(text_list, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def calculate_similarity(resume_embedding, job_embeddings):
    """
    Calculates the cosine similarity between a resume embedding and multiple job embeddings.

    Args:
        resume_embedding (np.ndarray): 1D array of the resume's BERT vector.
        job_embeddings (np.ndarray): 2D array of job BERT vectors.

    Returns:
        np.ndarray: 1D array of cosine similarity scores.
    """
    # Ensure the resume embedding is 2D for cosine_similarity
    resume_embedding_2d = resume_embedding.reshape(1, -1)
    
    # Cosine similarity is a common metric for text matching
    scores = cosine_similarity(resume_embedding_2d, job_embeddings)
    
    # scores is a 2D array [[score1, score2, ...]], so we flatten it to 1D
    return scores.flatten()

# --- Utility to prepare embeddings for MySQL BLOB storage ---
def embedding_to_bytes(embedding):
    """Converts a numpy embedding array to bytes for BLOB storage."""
    return embedding.tobytes()

def bytes_to_embedding(byte_data, dtype=np.float32):
    """Converts BLOB data back into a numpy embedding array."""
    # Assuming the Sentence-BERT model uses float32, which is typical
    return np.frombuffer(byte_data, dtype=dtype)

# Example to pre-calculate and save initial job embeddings
# job_descriptions = ["Python developer specializing in web frameworks.", "Financial analyst with strong Excel skills."]
# embeddings = get_embedding(job_descriptions)
# for emb in embeddings:
#     byte_data = embedding_to_bytes(emb)
#     # This byte_data is what you would store in the 'embedding' BLOB column for each job.
3. Flask Application (app.py)
This file sets up the API endpoint, connects to the MySQL database, and uses the model service to return job recommendations.

app.py

Python

from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
import model_service # Import the file we created above

app = Flask(__name__)

# --- MySQL Database Configuration (UPDATE WITH YOUR CREDENTIALS) ---
DB_CONFIG = {
    'user': 'your_mysql_user',        # e.g., 'root'
    'password': 'your_mysql_password',  # e.g., 'password123'
    'host': 'localhost',
    'database': 'job_recommender'
}

def get_db_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

# --- Recommendation API Endpoint ---
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # 1. Get resume text from the incoming JSON request
    data = request.get_json()
    resume_text = data.get('resume_text')
    top_n = data.get('top_n', 5) # Default to top 5 recommendations

    if not resume_text:
        return jsonify({"error": "Missing 'resume_text' in request body."}), 400

    # 2. Generate the BERT embedding for the user's resume text
    try:
        resume_embedding = model_service.get_embedding([resume_text])[0]
    except Exception as e:
        return jsonify({"error": f"BERT embedding failed: {e}"}), 500

    # 3. Fetch all jobs and their embeddings from MySQL
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed."}), 500

    cursor = conn.cursor()
    cursor.execute("SELECT job_id, title, description, embedding FROM job_listings")
    job_records = cursor.fetchall()
    cursor.close()
    conn.close()

    if not job_records:
        return jsonify({"message": "No job listings available."}), 200

    # Separate job metadata and embeddings
    job_metadata = []
    job_embeddings = []
    
    for job_id, title, description, embedding_bytes in job_records:
        job_metadata.append({'job_id': job_id, 'title': title, 'description': description})
        # Convert the BLOB data back to a numpy array
        job_embeddings.append(model_service.bytes_to_embedding(embedding_bytes))
        
    job_embeddings_matrix = np.array(job_embeddings)

    # 4. Calculate similarity scores
    similarity_scores = model_service.calculate_similarity(
        resume_embedding, 
        job_embeddings_matrix
    )

    # 5. Rank and sort results
    # Combine job metadata and scores
    scored_jobs = []
    for metadata, score in zip(job_metadata, similarity_scores):
        metadata['similarity_score'] = float(score)
        scored_jobs.append(metadata)

    # Sort by score in descending order and select the top N
    scored_jobs.sort(key=lambda x: x['similarity_score'], reverse=True)
    top_recommendations = scored_jobs[:top_n]

    # 6. Return the results
    return jsonify({
        "status": "success",
        "recommendations": top_recommendations
    })

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, port=5000)
4. How to Test the API
Run the Flask App:

Bash

python app.py
Send a POST Request:
Open a new terminal and use curl (or an application like Postman) to test the recommendation API:

Bash

curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d '{
    "resume_text": "I am a Python developer with 5 years of experience in data analysis and machine learning.",
    "top_n": 3
}'
The server will process the resume text, compare its BERT embedding to all the jobs in your MySQL database, and return the top 3 best-matching jobs.

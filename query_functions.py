import json
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
import torch
import gc

# Load the data with optimized memory usage
try:
    # Try to load the transformed dataset with optimized dtypes
    catalog_df = pd.read_csv("transformed_data1.csv", 
                           dtype={
                               'Assessment Name': 'category',
                               'Test Type': 'category',
                               'Remote Testing': 'category',
                               'Adaptive/IRT': 'category',
                               'Skills': 'category',
                               'Description': 'string',
                               'Relative URL': 'string'
                           })
    catalog_df.columns = catalog_df.columns.str.strip()
except FileNotFoundError:
    raise FileNotFoundError("transformed_data1.csv not found!")

def combine_row(row):
    """Combine row data into a single string for embedding."""
    parts = [
        str(row["Assessment Name"]),
        str(row["Duration in mins"]),
        str(row["Remote Testing"]),
        str(row["Adaptive/IRT"]),
        str(row["Test Type"]),
        str(row["Skills"]),
        str(row["Description"]),
    ]
    return ' '.join(parts)

# Prepare the dataset
catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
corpus = catalog_df['combined'].tolist()

# Initialize the model with memory optimization
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Using a smaller model
model.max_seq_length = 128  # Reduce sequence length

# Generate embeddings in batches to save memory
BATCH_SIZE = 32
corpus_embeddings = []
for i in range(0, len(corpus), BATCH_SIZE):
    batch = corpus[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, convert_to_tensor=True)
    corpus_embeddings.append(batch_embeddings)
    gc.collect()  # Force garbage collection

corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

def extract_url_from_text(text):
    """Extract URL from text if present."""
    match = re.search(r'(https?://[^\s,]+)', text)
    return match.group(1) if match else None

def extract_text_from_url(url):
    """Extract text content from a URL."""
    try:
        response = requests.get(url, headers={'User-Agent': "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(soup.get_text().split())
    except Exception as e:
        return f"Error: {e}"

def extract_features_from_query(query):
    """Extract key features from query using pattern matching."""
    features = {
        'skills': [],
        'duration': None,
        'test_type': None,
        'remote': None,
        'adaptive': None
    }
    
    # Extract duration
    duration_match = re.search(r'(\d+)\s*(?:min|minutes|mins)', query.lower())
    if duration_match:
        features['duration'] = int(duration_match.group(1))
    
    # Extract common skills
    skills = ['python', 'java', 'javascript', 'sql', 'problem solving', 'communication', 'teamwork']
    for skill in skills:
        if skill.lower() in query.lower():
            features['skills'].append(skill)
    
    # Extract test type
    test_types = ['coding', 'cognitive', 'personality', 'communication', 'aptitude']
    for test in test_types:
        if test.lower() in query.lower():
            features['test_type'] = test
            break
    
    # Check for remote/adaptive preferences
    if 'remote' in query.lower():
        features['remote'] = True
    if 'adaptive' in query.lower():
        features['adaptive'] = True
    
    # Convert features to search string
    search_terms = []
    if features['skills']:
        search_terms.extend(features['skills'])
    if features['test_type']:
        search_terms.append(features['test_type'])
    if features['duration']:
        search_terms.append(f"{features['duration']} minutes")
    if features['remote']:
        search_terms.append("remote testing")
    if features['adaptive']:
        search_terms.append("adaptive")
    
    return ' '.join(search_terms)

def filter_recommendations(recommendations, query):
    """Filter recommendations based on query constraints."""
    filtered = []
    
    # Extract duration constraint if any
    duration_match = re.search(r'(?:under|less than|within|max)\s*(\d+)\s*(?:min|minutes|mins)', query.lower())
    max_duration = int(duration_match.group(1)) if duration_match else None
    
    for rec in recommendations:
        # Apply duration filter if specified
        if max_duration and rec['Duration in mins'] and float(rec['Duration in mins']) > max_duration:
            continue
            
        # Check for remote testing requirement
        if 'remote' in query.lower() and rec['Remote Testing'].lower() != 'yes':
            continue
            
        # Check for adaptive testing requirement
        if 'adaptive' in query.lower() and rec['Adaptive/IRT'].lower() != 'yes':
            continue
        
        filtered.append(rec)
    
    return filtered if filtered else recommendations[:1]

def find_recommendations(query, k=5):
    """Find top-k recommendations for a query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Process in batches to save memory
    batch_size = 32
    top_scores = []
    top_indices = []
    
    for i in range(0, len(corpus_embeddings), batch_size):
        batch_embeddings = corpus_embeddings[i:i + batch_size]
        batch_scores = util.cos_sim(query_embedding, batch_embeddings)[0]
        batch_top_k = min(k, len(batch_scores))
        batch_top_scores, batch_top_indices = torch.topk(batch_scores, k=batch_top_k)
        
        top_scores.extend(batch_top_scores.tolist())
        top_indices.extend([idx + i for idx in batch_top_indices.tolist()])
        
        # Keep only top k overall
        if len(top_scores) > k:
            top_scores = sorted(top_scores, reverse=True)[:k]
            top_indices = [idx for _, idx in sorted(zip(top_scores, top_indices), reverse=True)][:k]
    
    results = []
    for score, idx in zip(top_scores, top_indices):
        result = {
            "Assessment Name": catalog_df.iloc[idx]['Assessment Name'],
            "Skills": catalog_df.iloc[idx]['Skills'],
            "Test Type": catalog_df.iloc[idx]['Test Type'],
            "Description": catalog_df.iloc[idx]['Description'],
            "Remote Testing": catalog_df.iloc[idx]['Remote Testing'],
            "Adaptive/IRT": catalog_df.iloc[idx]['Adaptive/IRT'],
            "Duration in mins": catalog_df.iloc[idx]['Duration in mins'],
            "Relative URL": catalog_df.iloc[idx]['Relative URL'],
            "Score": round(float(score), 4)
        }
        results.append(result)
    
    return results

def query_handling(query):
    """Main function to handle queries and return recommendations."""
    # Handle URL if present
    url = extract_url_from_text(query)
    if url:
        extracted_text = extract_text_from_url(url)
        query += " " + extracted_text
    
    # Extract features from query
    enhanced_query = extract_features_from_query(query)
    
    # Get initial recommendations
    recommendations = find_recommendations(enhanced_query if enhanced_query else query, k=10)
    
    # Filter recommendations based on constraints
    filtered_recommendations = filter_recommendations(recommendations, query)
    
    return pd.DataFrame(filtered_recommendations)

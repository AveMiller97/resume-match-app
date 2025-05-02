import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: download NLTK data
import nltk
import os
import nltk
import os

# Set a custom NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

# Download necessary NLTK data
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Tell NLTK where to look for data
nltk.data.path.append(nltk_data_path)

# Download NLTK data only if missing
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Set NLTK path so it can find the downloaded data
nltk.data.path.append(nltk_data_path)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to clean and tokenize text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # remove punctuation/numbers
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Compare resume and job description
def compare_texts(resume_text, job_description_text):
    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_description_text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_score = round(similarity * 100, 2)
    
    return match_score

# Example usage
resume = """Insert resume text here"""
job_description = """Insert job description text here"""

score = compare_texts(resume, job_description)
print(f"Match Score: {score}%")

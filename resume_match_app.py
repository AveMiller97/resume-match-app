import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Automatically download the punkt tokenizer and stopwords if not already available
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading the punkt tokenizer...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')

# Run the function to ensure resources are available
ensure_nltk_resources()

# Import NLTK components
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to clean and tokenize text
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation/numbers
    stop_words = set(stopwords.words('english'))  # Stopwords set
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)  # Return cleaned text as a string

# Compare resume and job description
def compare_texts(resume_text, job_description_text):
    # Preprocess both texts
    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_description_text)
    
    # Vectorize the cleaned texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
    
    # Compute cosine similarity between the two texts
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Calculate match score as a percentage
    match_score = round(similarity * 100, 2)
    
    return match_score

# Example usage
if __name__ == "__main__":
    resume = """Insert resume text here"""
    job_description = """Insert job description text here"""

    # Calculate and print match score
    score = compare_texts(resume, job_description)
    print(f"Match Score: {score}%")

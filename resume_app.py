import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib
import spacy
from collections import Counter

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load or Train Model
@st.cache_resource
def load_model():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df['Cleaned_Resume'] = df['Resume'].apply(lambda text: re.sub(r'\W', ' ', str(text).lower().strip()))
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['Cleaned_Resume'], df['Category'])
    joblib.dump(model, "resume_classifier.pkl")  # Save model for reuse
    
    return model

classifier = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_desc_vector], resume_vectors).flatten()
    return [round(score * 10, 2) for score in cosine_similarities]

# Function to suggest improvements using NLP
def suggest_improvements(resume_text):
    suggestions = []
    doc = nlp(resume_text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    word_freq = Counter(words)
    
    if word_freq["python"] == 0:
        suggestions.append("Consider adding Python as a skill if relevant.")
    if word_freq["teamwork"] == 0:
        suggestions.append("Highlight teamwork or collaboration experience.")
    if "projects" not in resume_text.lower():
        suggestions.append("Include details about projects you've worked on.")
    if len(resume_text.split()) < 150:
        suggestions.append("Your resume seems too short. Consider adding more details about your experience.")
    if "certification" not in resume_text.lower() and "course" not in resume_text.lower():
        suggestions.append("Mention relevant certifications or courses completed.")
    if word_freq["leadership"] == 0:
        suggestions.append("Showcase any leadership roles or responsibilities you've taken on.")
    if word_freq["achievements"] == 0 and word_freq["awards"] == 0:
        suggestions.append("Add any significant achievements or awards.")
    if word_freq["internship"] == 0 and word_freq["experience"] == 0:
        suggestions.append("Include details about past work experience or internships.")
    
    return suggestions if suggestions else ["Resume looks well-structured!"]

# Streamlit UI
st.title("AI Resume Screening & Ranking System")

st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)
    
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score (Out of 10)": scores})
    results = results.sort_values(by="Score (Out of 10)", ascending=False)
    st.write(results)
    
    st.header("Resume Analysis & Improvements")
    for i, file in enumerate(uploaded_files):
        st.subheader(f"Analysis for {file.name}")
        category = classifier.predict([resumes[i]])[0]
        st.write(f"**Predicted Job Category:** {category}")
        
        improvements = suggest_improvements(resumes[i])
        st.write("**Improvement Suggestions:**")
        for suggestion in improvements:
            st.write(f"- {suggestion}")
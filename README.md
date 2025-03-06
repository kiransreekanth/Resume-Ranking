# Resume-Ranking
This project is an AI-powered Resume Screening & Ranking System that evaluates resumes based on a given job description. It extracts resume text, processes it using Natural Language Processing (NLP), and applies machine learning techniques to classify and rank resumes. The system also provides improvement suggestions to enhance resume quality.

Features
✔ Extracts text from PDF resumes using PyPDF2,
✔ Cleans and preprocesses text (stopword removal, tokenization),
✔ Uses TF-IDF Vectorization for feature extraction,
✔ Predicts job category using a Naïve Bayes classifier,
✔ Computes cosine similarity to rank resumes,
✔ Suggests improvements using NLP-based keyword analysis,
✔ Provides an interactive UI using Streamlit

Technologies Used
1. Programming Language
Python 3.x

2. Machine Learning & NLP Libraries
Scikit-learn – TF-IDF vectorization, Naïve Bayes classification,
Spacy – NLP for resume analysis,
PyPDF2 – Extracting text from PDF resumes,
Joblib – Saving and loading trained models,
Pandas & NumPy – Data handling

3. Development Tools
Streamlit – UI development,
Flask (Optional) – Backend API deployment,
Jupyter Notebook / VS Code – Development environment

4. Dataset
Updated Resume Dataset (CSV Format) – Used for training the model

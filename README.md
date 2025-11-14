# 🚀 AI-Powered Job Recommendation System using RAG (FAISS + LangChain + Gemini)

A powerful AI-driven job recommendation system that uses *semantic search, **vector embeddings, and **Retrieval-Augmented Generation (RAG)* with *Google Gemini* to help users discover the most relevant job opportunities. The system intelligently analyzes user skills, job preferences, and optional resume input, retrieves the closest job matches using FAISS, and then asks Gemini to explain WHY those roles match (or don’t match) the user profile.

This project demonstrates real-world *AI Engineering, NLP, LLM Integration, modern UI development, and **ML deployment practices*.

---

## 🎯 Objective

To build a complete, intelligent job recommendation system that:
- Accepts user input (skills, location, job type)
- Uses *Sentence Transformers* to convert user profile to embeddings
- Retrieves semantically similar jobs using *FAISS Vector Search*
- Applies *RAG with Gemini* to explain each recommendation
- Presents results in an interactive *Streamlit UI*

---

#  Why This Is a RAG Project

This system performs the full RAG workflow:

### *1. Retrieve*
Using FAISS vector search, it retrieves the most relevant job postings.

### *2. Augment*
User profile + top job descriptions are combined to create a RAG prompt.

### *3. Generate*
Gemini generates:
- Why a job matches your skills
- Why it does NOT match (location mismatch, wrong job type, etc.)

This provides EXPLAINABLE AI (XAI), making your system smarter and more trustworthy.

---

# ✨ Features

### 🔍 1. Semantic Job Recommendations  
Uses vector similarity search instead of keyword search.

### 🤖 2. Gemini-Powered Explanations (RAG)  
Explains each recommendation with simple sentences:
- Why it matches skills  
- Why location/job type mismatch reduces suitability  

### 📄 3. Resume Upload Support  
Upload .pdf/.docx/.txt (extendable).

### 📊 4. Match Score with Progress Bar  
Score calculated from FAISS distances.

### 🎨 5. Beautiful Streamlit UI  
- Two-column layout  
- Gradient themes  
- Job cards  
- Professionally styled  

### 📥 6. Download Results  
Export recommendations as CSV.

---

📦 Tech Stack
Core

Python 3.10+

Streamlit

Sentence-Transformers

FAISS

Pandas, NumPy

RAG

LangChain

LangChain-Google-GenAI

Gemini API (Flash / Pro)

Extras

Virtualenv

GitHub

CSV
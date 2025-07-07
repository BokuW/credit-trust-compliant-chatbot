# CrediTrust Financial Complaint Chatbot

This repository contains the code and resources for building an intelligent complaint analysis chatbot for CrediTrust Financial, powered by Retrieval-Augmented Generation (RAG).

## Project Overview

The goal of this project is to transform raw, unstructured customer complaint data from various channels into actionable insights for internal stakeholders like Product Managers, Customer Support, and Compliance teams. The chatbot will allow users to ask plain-English questions about customer complaints and receive synthesized, evidence-backed answers.

## Project Structure

## Setup and Installation

(To be filled out later with steps to set up the environment and run the application)

## Progress

### Task 1: Exploratory Data Analysis and Data Preprocessing
- [ ] Load the full CFPB complaint dataset.
- [ ] Perform an initial EDA.
- [ ] Filter the dataset.
- [ ] Clean the text narratives.
- [ ] Deliverables: `notebooks/eda_preprocessing.ipynb`, `data/processed/filtered_complaints.csv`, EDA summary in `reports/eda_summary.md`.

### Task 2: Text Chunking, Embedding, and Vector Store Indexing
- [ ] Implement text chunking.
- [ ] Choose an embedding model.
- [ ] Generate embeddings and create a vector store.
- [ ] Deliverables: relevant scripts in `src/`, `vector_store/`, section in report.

### Task 3: Building the RAG Core Logic and Evaluation
- [ ] Implement Retriever.
- [ ] Design prompt template.
- [ ] Implement Generator.
- [ ] Perform qualitative evaluation.
- [ ] Deliverables: `src/rag_pipeline.py`, evaluation table in report.

### Task 4: Creating an Interactive Chat Interface
- [ ] Build UI with Gradio/Streamlit.
- [ ] Display sources and optionally implement streaming.
- [ ] Deliverables: `app.py`, screenshots/GIF in report
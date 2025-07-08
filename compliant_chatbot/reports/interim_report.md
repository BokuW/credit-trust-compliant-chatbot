# Interim Project Report: CrediTrust Financial Complaint Chatbot

## 1. Introduction and Project Overview

This interim report provides a summary of the initial phases of the CrediTrust Financial Complaint Chatbot project. The primary objective of this project is to develop a Retrieval-Augmented Generation (RAG) powered chatbot to enable efficient analysis of consumer complaints. By leveraging AI, the chatbot aims to transform raw, unstructured complaint data into actionable insights for internal stakeholders, allowing them to query complaint information using natural language and receive evidence-backed responses.

## 2. Project Setup and Environment

The project began with establishing a robust development environment and project structure:

* **Repository Setup:** A GitHub repository (`credit-trust-compliant-chatbot`) was initialized to manage source code, track changes, and facilitate version control.
* **Local Development Environment:** A Python virtual environment (`.venv`) was set up on the local machine using `py -3.11 -m venv .venv` to ensure dependency isolation. All required libraries are listed in `requirements.txt`.
* **Directory Structure:** A clear and organized directory structure was established, including `data/` (for raw and processed data), `notebooks/` (for exploratory work), `src/` (for modular Python code), `reports/` (for documentation and reports), `vector_store/` (for future vector database storage), and `models/` (for potential local models). Visual Studio Code is used as the primary IDE.

## 3. Task 1: Data Acquisition, EDA, and Preprocessing

Task 1 focused on understanding the structure, content, and quality of the CFPB complaint data and preparing it for the RAG pipeline.

### 3.1 Data Acquisition and Handling Large Files

The full CFPB complaint dataset, provided as a large CSV file (approx. 5.63 GB unzipped), posed significant challenges for local download and memory management. This was effectively addressed by:

* **Leveraging Google Colab:** The dataset was accessed directly from a shared Google Drive folder using Google Colab, eliminating the need for local download and benefiting from Colab's enhanced computational resources.
* **Selective Column Loading:** During the initial `pandas.read_csv` operation, only essential columns (`Product`, `Consumer complaint narrative`, `Company`, `Issue`, `Date received`) were loaded using the `usecols` parameter, significantly reducing memory consumption (from multi-GB to ~366 MB initially).

### 3.2 Exploratory Data Analysis (EDA) Findings

Initial EDA was performed on the loaded data to understand its characteristics:

* **Dataset Size:** The raw dataset contained over 9.6 million entries.
* **Product Distribution:** The 'Product' column showed a diverse range of financial products, with "Credit reporting or other personal consumer reports" being overwhelmingly the most frequent category.
* **Consumer Complaint Narrative Analysis:**
    * A critical finding was the high proportion of missing narratives: approximately **69% of the original 9.6 million complaints lacked a consumer narrative (6,629,041 out of 9,609,797 entries)**. These entries are unsuitable for the RAG pipeline and were targeted for removal.
    * For narratives that were present, the word count varied significantly, with an average length of about 54 words and some complaints being exceptionally detailed, extending up to 6,469 words.

### 3.3 Data Filtering and Cleaning

Based on the EDA, the dataset underwent targeted filtering and cleaning:

* **Product Filtering:** The dataset was filtered to include only complaints related to the five specified product categories: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers. It was observed that a distinct 'Buy Now, Pay Later (BNPL)' product category was **not explicitly present** in the 'Product' column of the raw data. Filtering for BNPL would require searching within narrative text or other columns, which will be implicitly handled by the RAG system's capabilities for broader semantic understanding. The filter relied on the closest available product classifications for the other four categories (e.g., 'Credit card' and 'Credit card or prepaid card' for Credit Card complaints).
* **Narrative Removal:** All records with missing (`NaN`) or effectively empty (whitespace-only) consumer complaint narratives were rigorously removed. This step was crucial given the high initial percentage of missing narratives.
* **Text Cleaning:** The 'Consumer complaint narrative' text underwent several normalization steps to prepare it for embedding:
    * All text was converted to lowercase.
    * Common boilerplate phrases (e.g., "I am writing to file a complaint...") were removed.
    * Sequences of 'x' (often used for PII redaction) were standardized to 'XX'.
    * Numbers and special characters were removed.
    * Excessive whitespace was consolidated, and leading/trailing spaces were stripped.

### 3.4 Final Dataset Status

After completing all filtering and cleaning operations, the refined dataset consists of **465,679 rows and 5 columns**. This cleaned and substantially reduced dataset is now stored as `filtered_complaints.csv` in the `data/processed/` directory on Google Drive, ready for the next stages of the RAG pipeline development.

## 4. Current Status and Next Steps

Task 1 is successfully completed, providing a high-quality, preprocessed dataset. The immediate next steps will involve **Task 2: Text Chunking, Embedding, and Vector Store Indexing**, where the cleaned narratives will be transformed into numerical representations and stored in a searchable vector database.
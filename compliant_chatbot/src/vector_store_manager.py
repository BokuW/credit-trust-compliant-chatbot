import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# --- Configuration for Chunking and Embedding ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_chunks(df: pd.DataFrame) -> list[Document]:
    """
    Chunks the 'Consumer complaint narrative' from the DataFrame into LangChain Documents.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Consumer complaint narrative' column
                          and a 'complaint_id' column (or similar unique ID).

    Returns:
        list[Document]: A list of LangChain Document objects, each representing a text chunk
                        with associated metadata.
    """
    print(f"Creating chunks with chunk_size={CHUNK_SIZE} and chunk_overlap={CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    documents = []
    for index, row in df.iterrows():
        narrative_content = str(row['Consumer complaint narrative'])
        # Ensure 'Issue' and 'Company' are strings to avoid potential errors in metadata
        issue_str = str(row.get('Issue', 'N/A'))
        company_str = str(row.get('Company', 'N/A'))
        date_received_str = str(row.get('Date received', 'N/A'))

        metadata = {
            'complaint_id': row['complaint_id'],
            'Product': row['Product'],
            'Issue': issue_str,
            'Company': company_str,
            'Date received': date_received_str # Include date received
        }
        documents.append(Document(page_content=narrative_content, metadata=metadata))

    all_chunks = text_splitter.split_documents(documents)
    print(f"Original documents: {len(documents)}")
    print(f"Total chunks created: {len(all_chunks)}")

    # Add chunk index to metadata for better traceability
    for i, chunk in enumerate(all_chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata['complaint_id']}-{i}"

    return all_chunks

def get_embedding_model():
    """
    Loads and returns the chosen Sentence Transformer embedding model.

    Returns:
        SentenceTransformerEmbeddings: The loaded embedding model.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return embeddings_model

def build_vector_store(chunks: list[Document], embeddings_model: SentenceTransformerEmbeddings, persist_directory: str):
    """
    Builds and persists a ChromaDB vector store from text chunks.

    Args:
        chunks (list[Document]): A list of LangChain Document objects (chunks).
        embeddings_model (SentenceTransformerEmbeddings): The embedding model to use.
        persist_directory (str): The directory where the ChromaDB will be saved.
    """
    print(f"Building and persisting ChromaDB vector store to: {persist_directory}")

    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    try:
        # Initialize ChromaDB from documents, generating embeddings
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )
        # Chroma.from_documents automatically persists if persist_directory is set
        print("ChromaDB vector store created and persisted successfully.")
    except Exception as e:
        print(f"An error occurred while creating or persisting the vector store: {e}")
        raise # Re-raise the exception to indicate failure

def load_vector_store(persist_directory: str, embeddings_model: SentenceTransformerEmbeddings):
    """
    Loads an existing ChromaDB vector store from disk.

    Args:
        persist_directory (str): The directory where the ChromaDB is saved.
        embeddings_model (SentenceTransformerEmbeddings): The embedding model used to create the store.

    Returns:
        Chroma: The loaded ChromaDB vector store.
    """
    print(f"Loading ChromaDB vector store from: {persist_directory}")
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model
        )
        print("ChromaDB vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error loading ChromaDB vector store: {e}")
        raise # Re-raise the exception

if __name__ == "__main__":
    # This block allows you to run this script directly for testing purposes
    # without needing the notebook. For actual project flow, you'll call these
    # functions from your notebook or app.py
    print("Running vector_store_manager.py directly for testing...")

    # --- Dummy Data for Testing ---
    data = {
        'Consumer complaint narrative': [
            "This is a long complaint about my credit card. I had unauthorized charges on XX. They debited my account unfairly.",
            "My personal loan application was denied without clear reason. I need to understand why. Company XX did not help.",
            "I noticed suspicious activity on my savings account. Money was transferred out without my consent on XX.",
            "This is another long complaint about banking fees. The charges were excessive and unexpected.",
            "A fraudster opened an account in my name. The company XX refused to close it even after I provided all documents."
        ],
        'Product': ['Credit card', 'Personal loan', 'Savings account', 'Checking or savings account', 'Credit card'],
        'Issue': ['Unauthorized transaction', 'Loan denied', 'Fraud', 'Fees', 'Identity theft'],
        'Company': ['ABC Bank', 'XYZ Lending', 'DEF Credit Union', 'GHI Financial', 'JKL Corp'],
        'Date received': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20']
    }
    test_df = pd.DataFrame(data)
    test_df['complaint_id'] = test_df.index # Add a dummy ID

    test_persist_dir = "./test_vector_store"
    if os.path.exists(test_persist_dir):
        import shutil
        shutil.rmtree(test_persist_dir) # Clean up previous test store

    try:
        test_chunks = create_chunks(test_df)
        test_embeddings_model = get_embedding_model()
        build_vector_store(test_chunks, test_embeddings_model, test_persist_dir)
        print(f"Test vector store built at {test_persist_dir}")

        # Test loading the store
        loaded_vectorstore = load_vector_store(test_persist_dir, test_embeddings_model)
        query = "fraudulent charges on my account"
        retrieved_docs = loaded_vectorstore.similarity_search(query, k=1)
        print(f"\nTest retrieval for '{query}':")
        for doc in retrieved_docs:
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error during direct script test: {e}")
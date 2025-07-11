## 5. Task 2: Text Chunking, Embedding, and Vector Store Indexing

This task focused on transforming the cleaned consumer complaint narratives into a semantically searchable format using embeddings and a vector database.

### 5.1 Text Chunking Strategy

To handle the variable lengths of complaint narratives and ensure effective embedding, a **`RecursiveCharacterTextSplitter`** from the LangChain library was employed. This splitter is chosen for its intelligent approach to maintaining semantic integrity by attempting to split documents first by larger, more meaningful delimiters (e.g., double newlines), then progressively smaller ones (single newlines, spaces, characters).

* **`chunk_size`:** An initial `chunk_size` of **500 characters** was selected. This size aims to capture sufficient context within each chunk without making the chunks excessively long, which could dilute the semantic meaning of their embeddings or exceed the context window of the downstream Language Model.
* **`chunk_overlap`:** A `chunk_overlap` of **50 characters** was used. This overlap ensures that the context from the end of one chunk is carried over to the beginning of the next. This prevents the loss of crucial information that might span across chunk boundaries and helps the retrieval process by providing overlapping semantic meaning.

This strategy ensures that queries can retrieve relevant snippets of information, even if the core answer is located between original document splits.

### 5.2 Embedding Model Choice

The **`sentence-transformers/all-MiniLM-L6-v2`** model was chosen for generating text embeddings.

* **Justification:** This model is widely recognized for its excellent balance of **efficiency, speed, and performance** across a broad spectrum of semantic similarity tasks. It produces compact embeddings (384 dimensions) which are computationally less intensive to store and compare, making it suitable for a dataset of this size within reasonable resource constraints. While larger models exist, `all-MiniLM-L6-v2` offers a robust general-purpose solution that provides good semantic representation for this application without requiring significant computational overhead, balancing accuracy with practicality.

### 5.3 Vector Store Creation and Persistence

A **ChromaDB** vector store was utilized to store the generated embeddings and their associated metadata.

* **Rationale for ChromaDB:** ChromaDB was selected for its ease of use, native persistence capabilities, and seamless integration with LangChain. It simplifies the process of managing embeddings and performing similarity searches.
* **Indexing Process:** For each text chunk, its vector embedding was generated using the chosen `all-MiniLM-L6-v2` model. Critically, essential metadata was stored alongside each vector, including:
    * `complaint_id` (derived from the original DataFrame index to uniquely identify the source complaint)
    * `Product`
    * `Issue`
    * `Company`
    * `Date received`
    This metadata is crucial for tracing retrieved chunks back to their original context and enriching the RAG pipeline's responses.
* **Persistence:** The entire vector store has been persisted to the `vector_store/` directory within the project structure on Google Drive, allowing for efficient loading and reuse without needing to re-embed the entire dataset.
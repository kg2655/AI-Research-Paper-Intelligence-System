from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentRAG:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        
    def ingest_document(self, chunks: list):
        """
        Embeds the document chunks and stores them in a FAISS index.
        """
        self.chunks = chunks
        if not chunks:
            return
            
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        # Inner Product with normalized vectors is equivalent to Cosine Similarity
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieves the top_k most relevant chunks for a given query.
        """
        if self.index is None or not self.chunks:
            return []
            
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                
        return results

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Builds a context-augmented prompt for the QA model.
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        context = "\n---\n".join(retrieved_chunks)
        
        prompt = (
            f"Answer the question based on the provided context.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt


import hashlib
import numpy as np
import pandas as pd
import os
from collections import Counter
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

class Preprocessing:
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.embedding_model.embed_query(text)
        return np.array(embedding).reshape(1, -1)

    def load_csv_documents(self, file_path: str) -> list[Document]:
        df = pd.read_csv(file_path)
        return [
            Document(page_content=str(row["text"]), metadata={"year": row.get("year", ""), "page": row.get("page", "")})
            for _, row in df.iterrows()
        ]

    def create_document_embeddings(self, docs: list[Document], embeddings_file: str):
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file, allow_pickle=True)
            if len(embeddings) == len(docs):
                return embeddings
            print("Embedding count mismatch. Regenerating.")

        embeddings = []
        for doc in docs:
            content = doc.page_content.strip()
            if content:
                embedding = np.array(self.embedding_model.embed_query(content)).reshape(1, -1)
                embeddings.append(embedding)

        np.save(embeddings_file, embeddings)
        return embeddings

    def generate_query_reformulations(self, query: str, n: int = 3) -> list[str]:
        prompt = (
            f"Rephrase the following question in {n} different ways from the perspective of a football fan asking about the FIFA World Cup. "
            f"Focus on football-related language, and keep the meaning the same. "
            f"List each rephrasing on a new line starting with a dash (-).\n\n"
            f"Original Question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled at rewriting questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            text = response.choices[0].message.content
            rephrasings = [q.strip("- ").strip() for q in text.strip().split("\n") if q.strip()]
            return rephrasings
        except Exception as e:
            print(f"Error generating query reformulations: {str(e)}")
            return []

    def rag_fusion_retrieval(self, query: str, docs, doc_embeddings, k: int = 10, n_reformulations: int = 3):
        queries = [query] + self.generate_query_reformulations(query, n=n_reformulations)
        doc_scores = Counter()
        doc_index_map = {}
        for q in queries:
            q_embedding = self.embed_query(q)
            for i, doc_embedding in enumerate(doc_embeddings):
                score = cosine_similarity(q_embedding, doc_embedding)[0][0]
                doc_scores[i] += score
                doc_index_map[i] = docs[i]
        top_indices = [idx for idx, _ in doc_scores.most_common(k)]
        top_docs = [doc_index_map[idx] for idx in top_indices]
        return top_docs

    def ask_groq_chatbot(self, query, top_k_docs, chat_history):
        context = "\n\n".join([doc.page_content for doc in top_k_docs])
        history_text = ""
        for i, (u, a) in enumerate(chat_history, 1):
            history_text += f"{i}. User: {u}\n   Assistant: {a}\n"
        history_text += f"{len(chat_history) + 1}. User: {query}"

        final_prompt = (
            "Answer the user's latest question strictly using the facts in the context below. "
            "Use the previous conversation history to resolve vague references like 'it' or 'they'. "
            "Do not explain your reasoning or state assumptions. "
            "If the answer is not in the context, say: 'I don't have that information, please visit www.footballhistory.org for more information.'\n\n"
            f"Context:\n{context}\n\nConversation History:\n{history_text}"
        )

        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error from Groq: {str(e)}")
            return "I'm having trouble answering that right now."

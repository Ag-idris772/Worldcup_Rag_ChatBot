# FIFA World Cup RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Groq for answering questions about FIFA World Cup history.  
This project scrapes historical data from [footballhistory.org](https://footballhistory.org), preprocesses it into clean, semantic chunks, and uses a custom RAG system to return grounded answers through a Streamlit interface.

---

## 📌 Project Stages

1. **Data Generation**  
   Scraped historical tournament pages from [footballhistory.org] using a CLI-based tool that extracts structured content (main sections and sidebar facts) into a CSV file.

2. **Data Cleaning & Preparation**  
   Applied semantic chunking using Sentence Transformers to group similar paragraphs. Enriched each chunk with metadata including year, host, chunk number, and document ID.

3. **Backend RAG System**  
   Built with an object-oriented architecture. Handles:  
   - Query embedding with `sentence-transformers/all-MiniLM-L6-v2`  
   - Reformulation of vague questions via Groq’s LLaMA 3  
   - RAG Fusion retrieval across multiple query rewrites  
   - Context-grounded answer generation

4. **Streamlit Interface**  
   A lightweight app with a chat-like experience. Supports:  
   - Interactive question answering  
   - Real-time reformulation and retrieval  
   - Chat history management

5. **Deployment**  
   Pushed to GitHub and deployed using Streamlit Community Cloud.

---

## 🔧 Tech Stack

- Python 3.11.7
- Streamlit
- LangChain
- Sentence Transformers
- Groq (LLaMA3-70B)
- BeautifulSoup (for scraping)
- HuggingFaceEmbeddings (for embedding retrieval)
- uv (for dependency resolution and locking)

---

## 📁 Dependency Management with uv

This project uses [uv](https://github.com/astral-sh/uv) — a super-fast Python package manager — to install and lock dependencies.

**To install uv:**
```sh
curl -Ls https://astral.sh/uv/install.sh | sh
```

---

## 🌐 Data Sources

- **[footballhistory.org](https://footballhistory.org)**  
  Comprehensive historical data on FIFA World Cup tournaments.

---

## ⚙️ Setup & Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/fifa-world-cup-rag-chatbot.git
    cd fifa-world-cup-rag-chatbot
    ```
2. Install `uv` as per the instructions above.
3. Install project dependencies:
    ```sh
    uv install
    ```
4. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

---

## 🛠️ Usage

- Ask questions about FIFA World Cup history in natural language.
- The chatbot will retrieve relevant information and provide grounded answers.
- Interact with the chat history to refine your queries or explore related topics.

---

## 📞 Contact

For any inquiries, please contact me through my linkedln [Idris Aguda](https://www.linkedin.com/in/idris-aguda-067630237?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B5heQJqZFRnOBxrAHbeVhbw%3D%3D).

---

## 📝 Acknowledgments

- Inspired by the need for accessible FIFA World Cup historical data.
- Leveraging cutting-edge AI and web technologies for innovative solutions.
- Couldn't have completed it without my good friend [Habeeb Oyewole](https://www.linkedin.com/in/habeeb-oyewole-690783294?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BNd1nimNURiewVYFmD3aFsA%3D%3D)

---

## 🛡️ Disclaimer

This project is not affiliated with or endorsed by FIFA or any related entities. It is an independent initiative for educational and informational purposes.

---

## 📚 References

- [LangChain Documentation](https://langchain.readthedocs.io/en/latest/)
- [Groq Documentation](https://docs.groq.co/)
- [Streamlit Documentation](https://docs.streamlit.io/library)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Sentence Transformers Documentation](https://www.sbert.net/docs/)
- [HuggingFaceEmbeddings Documentation](https://huggingface.co/docs/transformers/model_doc/auto)
- [uv Documentation](https://github.com/astral-sh/uv#readme)

---

## 🏁 Final Words

Thank you for exploring the FIFA World Cup RAG Chatbot project! We hope it serves as a valuable resource and inspires further innovations in the realm of AI and web development.


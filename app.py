import streamlit as st
import os
from backend import Preprocessing

# Set Streamlit page configuration
st.set_page_config(page_title="World Cup RAG Bot", page_icon="🏆", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button {
        background-color: #0455A4;
        color: grey;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #033e7b;
        cursor: pointer;
    }
    .chat-bubble {
        background-color: white;
        color: black; /* <-- Added this line to fix mobile dark mode issue */
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 1em;
        box-shadow: 0 1px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Display header and image
st.image("wc_image.jpg")  # Make sure this image exists in your project directory and is pushed to GitHub
st.title("FIFA World Cup Chatbot 🌎")
st.markdown("Ask anything about the FIFA World Cup and get smart, document-grounded answers!")

# Load backend processor
processor = Preprocessing()

# Load documents and embeddings (cached)
@st.cache_resource
def load_resources():
    file_path = "chunked_worldcup_data.csv"
    docs = processor.load_csv_documents(file_path)
    embeddings_file = "worldcup_embeddings.npy"
    doc_embeddings = processor.create_document_embeddings(docs, embeddings_file)
    return docs, doc_embeddings

docs, doc_embeddings = load_resources()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input and Ask button
user_query = st.text_input("Ask your question:", placeholder="e.g. Who won the 2022 world cup?")
submit_button = st.button("Ask", type="primary")

# Process input if button is clicked
if submit_button and user_query:
    with st.spinner("Thinking..."):
        reformulations = processor.generate_query_reformulations(user_query)
        top_k_docs = processor.rag_fusion_retrieval(user_query, docs, doc_embeddings)
        answer = processor.ask_groq_chatbot(user_query, top_k_docs, st.session_state.chat_history)
        st.session_state.chat_history.append((user_query, answer))

        # # Show top k retrieved documents (You can uncomment this section if you want to display retrieved documents in the UI)
        # st.markdown("---")
        # st.markdown("**Top Retrieved Documents:**")
        # for i, doc in enumerate(top_k_docs, 1):
        #     preview = doc.page_content.strip().replace("\n", " ")
        #     st.markdown(f"{i}. *{preview[:200]}{'...' if len(preview) > 200 else ''}*")       

# Display chat history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"<div class='chat-bubble'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble'><b>Idris:</b> {a}</div>", unsafe_allow_html=True)

# Footer
st.caption("© World Cup Chat Bot | Built with Groq, Streamlit, and LangChain | all data scraped from www.footballhistory.org")

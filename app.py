import streamlit as st
from backend import Preprocessing

st.set_page_config(page_title="World Cup RAG Bot", page_icon="🏆", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton>button { background-color: #0455A4; color: white; border-radius: 8px; padding: 0.5em 1em; }
    .chat-bubble { background-color: white; padding: 1em; border-radius: 10px; margin-bottom: 1em; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

st.image("C:\\Users\\aguda\\OneDrive\\Desktop\\wc_project\\wc image.jpg")
st.title("FIFA World Cup Chatbot 🌎")
st.markdown("Ask anything about the FIFA World Cup and get smart, document-grounded answers!")

processor = Preprocessing()

@st.cache_resource
def load_resources():
    file_path = "chunked_worldcup_data.csv"
    docs = processor.load_csv_documents(file_path)
    embeddings_file = "worldcup_embeddings.npy"
    doc_embeddings = processor.create_document_embeddings(docs, embeddings_file)
    return docs, doc_embeddings

docs, doc_embeddings = load_resources()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question:", placeholder="e.g. Who won the 2022 world cup?")

if query:
    with st.spinner("Thinking..."):
        # Always generate reformulations (no vague/clear classification)
        reformulations = processor.generate_query_reformulations(query)

        # Pass reformulations into retrieval
        top_k_docs = processor.rag_fusion_retrieval(query, docs, doc_embeddings)
        answer = processor.ask_groq_chatbot(query, top_k_docs, st.session_state.chat_history)

        st.session_state.chat_history.append((query, answer))

        # # Show top k retrieved documents (You can uncomment this section if you want to display retrieved documents in the UI)
        # st.markdown("---")
        # st.markdown("**Top Retrieved Documents:**")
        # for i, doc in enumerate(top_k_docs, 1):
        #     preview = doc.page_content.strip().replace("\n", " ")
        #     st.markdown(f"{i}. *{preview[:200]}{'...' if len(preview) > 200 else ''}*")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"<div class='chat-bubble'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble'><b>Idris:</b> {a}</div>", unsafe_allow_html=True)

st.caption("© World Cup Chat Bot | Built with Groq, Streamlit, and LangChain | all data scraped from www.footballhistory.org")

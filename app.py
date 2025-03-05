import streamlit as st
import chromadb
import utils

# Connect to database
@st.cache_resource
def init_db():
    try:
        chroma_client = chromadb.PersistentClient(path="./db")
        collection = chroma_client.get_collection("knowledge_base")
        print("Connected to existing ChromaDB collection")
        return collection
    except:
        print("Error: DB not found. Please run pipeline.py first.")
        return None

collection = init_db()

# App
st.title("Genshin Impact Agent")
st.header("Ask a Question")
col1, col2 = st.columns([3, 1])
query = st.text_input("Your question:",)
submit_button = st.button("Submit", type="primary")
if submit_button and query:
    with st.status("Generating response...") as status:
        # Process
        data = collection.get(include=["documents", "embeddings"])
        top_documents, top_scores, query_embedding = utils.semantic_search(data["documents"], data["embeddings"], query)
        context = utils.format_context(top_documents, top_scores)
        response = utils.generate_response(context, query)

        # Display
        status.update(label="Done!", state="complete")
            
        st.subheader("Response")
        st.write(response)
        
        st.subheader("Retrieved Context")
        st.write(context)
elif submit_button and not query:
    st.warning("Please enter a question before submitting.")
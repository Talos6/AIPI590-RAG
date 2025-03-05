__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
st.sidebar.title("Genshin Impact Agent")
page = st.sidebar.radio('Go To', ["Application", "About", "Details"])

if page == "Application":
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

elif page == "About":
    st.write("### Intro")
    st.write("Genshin impact agent is a AI assistant that can help you to answer questions about Genshin Impact game. Built on top of GPT and RAG.")
    st.write("### Problem")
    st.write("Consumer would be players who want to know details about this specific domain and get a accurate instant response.")
    st.write("### Solution")
    st.write("An AI assistant utilize the Genshin Impact Fandom wiki data as context.")
    st.write("### Usage")
    st.markdown("""
    - Ask any thing about Genshin Impact.
    - Check the response.
    - Also include the context for reference and more details.
    """)
    st.write("### Author")
    st.write("Xinyue(Yancey) Yang")
    st.write("### Repository")
    st.write("[Github link](https://github.com/Talos6/AIPI590-RAG)")
    st.write("### Instructions")
    code = """
        # Clone repo and ensure python and pip have installed
        git clone https://github.com/Talos6/AIPI590-RAG.git 

        # Install required libraries
        pip install -r requirements.txt

        # Run the application with trained model
        streamlit run main.py

        # Run the script to reset db
        python pipeline.py
    """
    st.code(code, language='bash')

elif page == "Details":
    # Details
    st.write("### Data Source")
    st.write("The data source is from Genshin Impact Fandom Wiki. [Wiki Link](https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki)")
    st.write("### Data Preprocessing")
    st.write("The data is formatted into a customized text, not web scrapped from the website directly to fully control the data quality.")
    st.write("### Data Pipeline")
    st.markdown("""
    - **Text Collection**: Text data are loaded from the .txt source files.
    - **Tokenization**: Sentences retrieved by tokenizing text through nltk.
    - **Chunk Data**: Put the sentences into chunks with a maximum of 500 words.
    - **Embedding**: Embedding the chunks with a pre-trained openai text-embedding-ada-002 model.
    - **Storage**: Stored documents and embeddings into ChromaDB for consistency.
    """)
    st.write("### Model")
    st.markdown("""
    - **Embedding Model**: OpenAI's text-embedding-ada-002 model is used to get the embeddings.
    - **Large Language Model**: OpenAI's GPT3.5-turbo model is used to generate responses.
    """)
    st.write("### Process")
    st.markdown("""
    - **Embed query**: Get embeddings for the user query.
    - **Retrive knowledge**: Retrive all knowledge from DB.
    - **Semantic Search**: Calculate similarity scores, and response top 5 documents as context.
    - **Prepare Prompt**: Prepare the final prompt with context and query.
    - **Generate Response**: Generate response using GPT3.5-turbo model.
    - **Display**: Display the response and context.
    """)
    st.write("### Evaluation")
    st.markdown("""
        - **Accuracy**: The response accuracy is evaluated by the response through user feedback. (90%)
        - **Relevance**: Measured by the response and context relevance through user feedback. (95%)
    """)
    st.write("### Future Work")
    st.markdown("""
    - **Automatic Data Collection**: Need to improve and set auto scheduler to web-scrap new data.
    - **Data Processing**: Data needs to be cleaned and in known words when web-scrapped.
    - **Monitors and Metrics**: Need to set up monitoring and metrics for the application.
    - **Feedback System**: Implement feedback system to improve the model.
    - **Model Evaluation**: Come up with other evaluation metrics for the model except derived from user-feedback.
    """)

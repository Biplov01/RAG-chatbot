# Import necessary libraries
import streamlit as st
import openai
from brain import get_index_for_pdf

# Set the title for the Streamlit app
st.title("Everything about myself")

# Set OpenAI API key directly (not recommended for production)
openai.api_key = "RAG Chatbot"

# Cached function to create a vectordb for the provided PDF files
@st.cache_data
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database creation in progress..."):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence is the context of the PDF extract with metadata. 
    Carefully focus on the metadata, especially 'filename' and 'page', whenever answering.
    
    Make sure to add filename and page number at the end of the sentence you are citing to.
        
    Reply "Not applicable" if the text is irrelevant.
     
    The PDF content is:
    {pdf_extract}
"""

# Initialize the chat history in session state if not already set
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

# Display previous chat messages
for message in st.session_state["prompt"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF.")
        st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the system prompt with the pdf extract
    st.session_state["prompt"][0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    st.session_state["prompt"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call OpenAI API with streaming and display the response
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=st.session_state["prompt"], stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt and update session state
    st.session_state["prompt"].append({"role": "assistant", "content": result})

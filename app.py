# Import necessary libraries
import streamlit as st
import openai
from brain import get_index_for_pdf
from io import BytesIO

# Set the title for the Streamlit app
st.title("Biplov RAG Chatbot")

# Create a text input for the OpenAI API key with a key emoji
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", placeholder="🔑 API Key")

# Path to the default PDF file
default_pdf_path = "a1r.pdf"

# Check if the API key is provided
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Cached function to create a vectordb for the provided PDF file
@st.cache_data
def create_vectordb(pdf_path):
    # Read the PDF file as bytes
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database creation in progress..."):
        vectordb = get_index_for_pdf([pdf_bytes], [pdf_path], openai.api_key)
    return vectordb

# Create the vectordb using the default PDF file
vectordb = create_vectordb(default_pdf_path)
st.session_state["vectordb"] = vectordb

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

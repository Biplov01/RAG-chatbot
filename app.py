# Import necessary libraries
import streamlit as st
from brain import get_index_for_pdf

# Set the title for the Streamlit app
st.title("Everything about Myself")

# Function to read the predefined PDF file (a1r.pdf)
def read_predefined_pdf():
    # Open the PDF file and return its content
    with open("a1r.pdf", "rb") as f:  # No path needed since it's in the same directory
        pdf_content = f.read()
    return pdf_content

# Cached function to create a vectordb for the predefined PDF file
@st.cache_data
def create_vectordb():
    # Read the predefined PDF file
    pdf_content = read_predefined_pdf()
    pdf_file_name = "a1r.pdf"
    
    # Show a spinner while creating the vectordb
    with st.spinner("Creating vector database from the PDF..."):
        vectordb = get_index_for_pdf([pdf_content], [pdf_file_name])  # Removed openai.api.key
    return vectordb

# Create the vectordb when the app starts
vectordb = create_vectordb()

# Initialize the chat history in session state if not already set
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

# Display previous chat messages
for message in st.session_state["prompt"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything about me")

# Handle the user's question
if question:
    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the system prompt with the PDF extract
    st.session_state["prompt"][0] = {
        "role": "system",
        "content": f"""
            You are a helpful Assistant who answers users' questions based on multiple contexts given to you.

            Keep your answer short and to the point.

            The evidence is the context of the PDF extract with metadata. 
            Carefully focus on the metadata, especially 'filename' and 'page', whenever answering.

            Make sure to add filename and page number at the end of the sentence you are citing to.
            
            Reply "Not applicable" if the text is irrelevant.
             
            The PDF content is:
            {pdf_extract}
        """,
    }

    # Add the user's question to the prompt and display it
    st.session_state["prompt"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Simulate the response process without using OpenAI API
    # Here you might implement your own response logic based on `pdf_extract`
    # For demonstration purposes, we will simply echo the question
    response_text = f"You asked: {question} (This is a placeholder response.)"
    
    # Write the response to the chat message
    botmsg.write(response_text.strip())

    # Add the assistant's response to the prompt and update session state
    st.session_state["prompt"].append({"role": "assistant", "content": response_text.strip()})

import openai
import streamlit as st

from brain import OPENROUTER_BASE_URL, get_index_for_pdf

# Any OpenRouter model ID works here
CHAT_MODEL = "openai/gpt-4o-mini"

# Set the title for the Streamlit app
st.title("Biplov RAG Chatbot")

# Add a large chatbot emoji below the title
st.markdown(
    "<h1 style='text-align: center; font-size: 100px;'>🤖</h1>",
    unsafe_allow_html=True
)

# Create a text input for the OpenRouter API key with a key emoji
openrouter_api_key = st.text_input("Enter your OpenRouter API Key", type="password", placeholder="🔑 API Key")

# Path to the default PDF file
default_pdf_path = "about2.pdf"

# Check if the API key is provided
if not openrouter_api_key:
    st.warning("Please enter your OpenRouter API key to proceed.")
    st.stop()

# OpenRouter is OpenAI-compatible, so the OpenAI client works with a different base URL
client = openai.OpenAI(
    api_key=openrouter_api_key,
    base_url=OPENROUTER_BASE_URL,
    max_retries=5,  # back off and retry on 429s
)


# Cached per PDF path and API key, so one user's key is never reused for another
@st.cache_resource(show_spinner="Vector database creation in progress...")
def create_vectordb(pdf_path, api_key):
    # Read the PDF file as bytes
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return get_index_for_pdf([pdf_bytes], [pdf_path], api_key)


# Create the vectordb using the default PDF file
vectordb = create_vectordb(default_pdf_path, openrouter_api_key)

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
    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)

    # Include filename and page so the model can cite them
    pdf_extract = "\n\n".join(
        f"[filename: {doc.metadata['filename']}, page: {doc.metadata['page']}]\n{doc.page_content}"
        for doc in search_results
    )

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

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=st.session_state["prompt"],
            stream=True
        )

        result = ""
        for chunk in response:
            # Some stream chunks carry no choices (e.g. usage data)
            if chunk.choices and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
                botmsg.write(result.strip())
    except openai.APIError as e:
        # Drop the unanswered question and show OpenRouter's actual reason
        st.session_state["prompt"].pop()
        botmsg.error(f"OpenRouter request failed: {e.message}")
        st.stop()

    # Add the assistant's response to the prompt and update session state
    st.session_state["prompt"].append({"role": "assistant", "content": result.strip()})

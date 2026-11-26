import re
from io import BytesIO
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """Extracts text from a PDF file and returns cleaned text."""
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        # Keep empty pages as "" so page numbers stay correct
        text = page.extract_text() or ""

        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Fix hyphenated words
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())  # Merge line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize multiple newlines
        output.append(text)
    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    """Converts extracted text into LangChain Document objects."""
    if isinstance(text, str):
        text = [text]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=0,
    )

    doc_chunks = []
    for page_number, page_text in enumerate(text, start=1):
        for i, chunk in enumerate(text_splitter.split_text(page_text)):
            doc_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_number,
                        "chunk": i,
                        "source": f"{page_number}-{i}",
                        "filename": filename,
                    },
                )
            )
    return doc_chunks


def docs_to_index(docs: List[Document], api_key: str) -> FAISS:
    """Creates a FAISS vector index from documents using OpenRouter embeddings."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,  # send raw text, not OpenAI token IDs
    )
    return FAISS.from_documents(docs, embeddings)


def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str], api_key: str) -> FAISS:
    """Processes multiple PDFs and returns a FAISS index."""
    if not pdf_files:
        raise ValueError("No PDF files provided")

    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))

    if not documents:
        raise ValueError("No extractable text found. Scanned PDFs need OCR first.")

    return docs_to_index(documents, api_key)

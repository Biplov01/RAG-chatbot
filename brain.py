import databutton as db
import re
from io import BytesIO
from typing import Tuple, List
import pickle

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import faiss


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """Extracts text from a PDF file and returns cleaned text."""
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if text is None:
            continue  # Skip empty pages

        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Fix hyphenated words
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())  # Merge line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize multiple newlines
        output.append(text)
    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    """Converts extracted text into LangChain Document objects."""
    if isinstance(text, str):
        text = [text]

    page_docs = [Document(page_content=page, metadata={"page": i + 1}) for i, page in enumerate(text)]

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=0,
    )

    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source": f"{doc.metadata['page']}-{i}",
                    "filename": filename,
                },
            )
            doc_chunks.append(chunk_doc)
    return doc_chunks


def docs_to_index(docs: List[Document], openai_api_key: str) -> FAISS:
    """Creates a FAISS vector index from documents using OpenAI embeddings."""
    return FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))


def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str], openai_api_key: str) -> FAISS:
    """Processes multiple PDFs and returns a FAISS index."""
    if not pdf_files:
        raise ValueError("No PDF files provided")

    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))

    return docs_to_index(documents, openai_api_key)

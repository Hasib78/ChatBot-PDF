import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # Updated import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Call load_dotenv here so it works on your local PC
load_dotenv()

PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_vectorstore(uploaded_files):
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename

        # FIX: Read the content ONCE and write it
        content = file.file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # IMPORTANT: Reset the cursor if you plan to use 'file' again,
        # though writing to disk and then loading via path is safer.
        file_paths.append(str(save_path))

    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # Use the API-based embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L12-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR, embedding_function=embeddings
        )
        vectorstore.add_documents(texts)
    else:
        vectorstore = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=PERSIST_DIR
        )

    return vectorstore


##OLD for sentence transformer:
# import os
# from pathlib import Path
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# PERSIST_DIR = "./chroma_store"
# UPLOAD_DIR = "./uploaded_pdfs"

# os.makedirs(UPLOAD_DIR, exist_ok=True)


# def load_vectorstore(uploaded_files):
#     file_paths = []

#     for file in uploaded_files:
#         save_path = Path(UPLOAD_DIR) / file.filename
#         with open(save_path, "wb") as f:
#             f.write(file.file.read())
#             file_paths.append(str(save_path))

#     docs = []

#     for path in file_paths:
#         loader = PyPDFLoader(path)
#         docs.extend(loader.load())

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = splitter.split_documents(docs)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

#     if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
#         vectorstore = Chroma(
#             persist_directory=PERSIST_DIR, embedding_function=embeddings
#         )
#         vectorstore.add_documents(texts)
#         vectorstore.persist()

#     else:
#         vectorstore = Chroma.from_documents(
#             documents=texts, embedding=embeddings, persist_directory=PERSIST_DIR
#         )
#         vectorstore.persist()

#     return vectorstore

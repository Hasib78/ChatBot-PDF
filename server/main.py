import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.load_vectorstore import load_vectorstore, PERSIST_DIR
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger


app = FastAPI(title="PdfChatbot")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"received {len(files)} files")

        load_vectorstore(files)

        logger.info("documents added to chroma")
        return {"message": "Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        )

        chain = get_llm_chain(vectorstore)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result
    except Exception as e:
        logger.exception("error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/clear_db/")
async def clear_database():
    try:
        logger.info("Clearing database and uploaded PDF files...")

        from modules.load_vectorstore import PERSIST_DIR, UPLOAD_DIR

        try:

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
            temp_vectorstore = Chroma(
                persist_directory=PERSIST_DIR, embedding_function=embeddings
            )
            temp_vectorstore.delete_collection()
            logger.info("ChromaDB collection wiped successfully.")
        except Exception as e:

            logger.warning(f"Note on Chroma wipe: {e}")

        if os.path.exists(UPLOAD_DIR):
            # loop through and delete just the files inside, leaving the folder intact
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # os.unlink safely deletes a single file
                except Exception as e:
                    logger.warning(
                        f"Failed to delete {file_path}. It might be open in another program: {e}"
                    )

        # Ensure the folder still exists just in case
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        logger.info("Full reset complete.")
        return {"message": "Database memory and PDF files cleared successfully!"}

    except Exception as e:
        logger.exception("Error during full database clear")
        return JSONResponse(status_code=500, content={"error": str(e)})

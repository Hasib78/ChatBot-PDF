import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def get_llm_chain(vectorstore):
    # Initialize the LLM
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

    # k=3 means it grabs the top 3 relevant PDF chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an expert assistant. Use the provided context from the PDFs "
        "to answer the user's question. If the answer isn't in the context, "
        "honestly say you don't know. Keep it professional.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the 'Stuff' Chain (Combines docs into the prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # links the retriever (Chroma) to the doc chain (LLM)
    return create_retrieval_chain(retriever, combine_docs_chain)

from logger import logger


def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running modern chain for input: {user_input}")

        # use .invoke() and pass "input" instead of "query"
        result = chain.invoke({"input": user_input})

        # grab "answer" and "context" from the new dictionary format
        response = {
            "response": result["answer"],
            "sources": [
                doc.metadata.get("source", "Unknown Source")
                for doc in result["context"]
            ],
        }

        logger.debug(f"Chain response: {response}")
        return response

    except Exception as e:
        logger.exception("Error in query_chain")
        raise

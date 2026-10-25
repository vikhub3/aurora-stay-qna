from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 5, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:
    prompt = f"""You are an AI assistant answering questions about AuroraStay hotel policies.

User Query:
{query}

Retrieved Policy Information:
{retrieved_chunks}

Rules:
1. Provide a direct, highly detailed, and complete answer based on the retrieved policy information.
2. Do not restrict the response to a single sentence; use multiple sentences, bullet points, or structured lists to ensure the response is clear, accurate, and completely addresses all parts of the user query.
3. Tables are represented with pipes (|). Read headers to locate matching values.
4. Use only information explicitly supported by the retrieved policy information.
5. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
6. Include all relevant information needed to answer the query completely (such as specific fees, conditions, policy codes, names of properties, exceptions, etc.).
7. Focus only on information directly relevant to the user's query.
8. Answer the user's specific question directly and ensure the response addresses what was asked.
9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or policy details.
10. If the retrieved information does not contain enough details to confidently answer the query, clearly state:
    "The available policy documents do not contain enough information to answer this question."
"""
    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model using the securely injected OpenAI API Key via environment variables
    model = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
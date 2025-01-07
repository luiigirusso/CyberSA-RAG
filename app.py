import os
import streamlit as st
from weaviate.classes.query import MetadataQuery
from langchain_openai import ChatOpenAI
import weaviate
from weaviate.classes.init import Auth
import textwrap

# Funzione principale del sistema RAG
def connect_to_weaviate():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_TOKEN"]
        }
    )
    return client


def similarity_search(client, question: str):
    jeopardy = client.collections.get("Triple")
    response = jeopardy.query.near_text(
        query=question,
        limit=4,
        return_metadata=MetadataQuery(distance=True)
    )
    return response.objects


def generate_RAG_answer(question: str, context: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-3.5-turbo"
    )
    prompt = f"Answer the question based on the context: \n\nContext: {context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return response.content


def generate_LLM_answer(question: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-3.5-turbo"
    )
    response = llm.invoke(question)
    return response.content


def main():
    st.title("RAG Chatbot with Weaviate and OpenAI")
    st.write("Ask your question and receive answers from both RAG and LLM!")

    client = connect_to_weaviate()

    if not client.is_ready():
        st.error("Weaviate client is not ready. Check your configuration.")
        return

    # Input area for the user question
    question = st.text_input("Enter your question:", "")

    if st.button("Ask"):
        if question.strip():
            # Perform similarity search
            st.subheader("Similarity Search Results:")
            context_objects = similarity_search(client, question)
            context = "\n".join([f"{o.properties} (distance: {o.metadata.distance})" for o in context_objects])
            st.text(context)

            # Generate RAG-based answer
            st.subheader("RAG Answer:")
            rag_answer = generate_RAG_answer(question, context)
            st.text(textwrap.fill(rag_answer, 60))

            # Generate LLM-based answer
            st.subheader("LLM Answer:")
            llm_answer = generate_LLM_answer(question)
            st.text(textwrap.fill(llm_answer, 60))
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

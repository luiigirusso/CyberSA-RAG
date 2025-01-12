import os
import streamlit as st
from weaviate.classes.query import MetadataQuery
from langchain_openai import ChatOpenAI
import weaviate
from weaviate.classes.init import Auth
import textwrap
from dotenv import load_dotenv

# Funzione principale del sistema RAG
def connect_to_weaviate():
    weaviate_url = os.getenv("WEAVIATE_URL_ARCH")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY_ARCH")

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
    load_dotenv("C:/Users/luigi/Desktop/TESI/repo/CyberSA-RAG/pipeline_without/.env", override=True)
    st.title("RAG Chatbot with Weaviate and OpenAI")
    st.write("Ask your question and get answers from both RAG and LLM!")

    client = connect_to_weaviate()

    if not client.is_ready():
        st.error("Weaviate client is not ready. Check your configuration.")
        return

    # Input area for the user question
    question = st.text_input("Enter your question:", "")

    # Inizializza lo stato della sessione per i dati
    if "context" not in st.session_state:
        st.session_state["context"] = ""
    if "rag_answer" not in st.session_state:
        st.session_state["rag_answer"] = ""
    if "llm_answer" not in st.session_state:
        st.session_state["llm_answer"] = ""
    if "show_similarity" not in st.session_state:
        st.session_state["show_similarity"] = False

    if st.button("Ask"):
        if question.strip():
            # Effettua la ricerca di similarità
            context_objects = similarity_search(client, question)
            context = "\n".join([f"{o.properties} (distance: {o.metadata.distance})" for o in context_objects])
            st.session_state["context"] = context

            # Genera la risposta RAG
            st.session_state["rag_answer"] = generate_RAG_answer(question, context)

            # Genera la risposta LLM
            st.session_state["llm_answer"] = generate_LLM_answer(question)

            # Resetta la visualizzazione della similarity search
            st.session_state["show_similarity"] = False
        else:
            st.warning("Please enter a question.")

    # Mostra le risposte
    if st.session_state["rag_answer"] or st.session_state["llm_answer"]:
        st.subheader("RAG Answer:")
        st.text(textwrap.fill(st.session_state["rag_answer"], 60))

        st.subheader("LLM Answer:")
        st.text(textwrap.fill(st.session_state["llm_answer"], 60))

        # Bottone per mostrare/nascondere i risultati della similarity search
        if st.button("Show Similarity Search Results"):
            st.session_state["show_similarity"] = not st.session_state["show_similarity"]

        # Mostra i risultati della similarity search se attivati
        if st.session_state["show_similarity"]:
            st.subheader("Similarity Search Results:")
            st.text(st.session_state["context"])


if __name__ == "__main__":
    main()

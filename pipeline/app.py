import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import textwrap
import numpy as np
from numpy.linalg import norm
import json
import streamlit as st
from dotenv import load_dotenv

# Funzione per caricare i dati JSON
def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))


# Funzione per fare la similarity search
def similarity_search(question):
    # Carica gli embedding dal file JSON
    embeddings_data = load_embeddings("pipeline/vector_database/entity_embeddings_arch.json")
    embeddings_data.extend(load_embeddings("pipeline/vector_database/relation_embeddings_arch.json"))

    embedding_model = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model="text-embedding-ada-002",
    )
    query_vector = embedding_model.embed_query(question)

    # Calcola la similarità per ciascun embedding
    similarities = []
    for entry in embeddings_data:
        entity_name = entry['name']
        embedding = np.array(entry['embedding'])
        similarity = cosine_similarity(query_vector, embedding)
        similarities.append((entity_name, similarity))

    # Ordina per similarità decrescente
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Restituisci i top_k risultati
    return similarities[:5]


# Funzione per generare una risposta dal modello LLM
def generate_RAG_answer(question: str, context: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-3.5-turbo"
    )
    prompt = f"Answer the question based on the context: \n\nContext: {context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return response.content

    # Funzione per generare una risposta dal modello LLM
def generate_LLM_answer(question: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-3.5-turbo"
    )
    response = llm.invoke(question)
    return response.content

def main():
    load_dotenv("C:/Users/luigi/Desktop/TESI/repo/CyberSA-RAG/pipeline/.env", override=True)
    st.title("RAG Chatbot with TransE embeddings")
    st.write("Ask your question and get answers from both RAG and LLM!")

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
            context = similarity_search(question)
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
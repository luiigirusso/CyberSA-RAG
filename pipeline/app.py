import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import textwrap
import numpy as np
import re
from numpy.linalg import norm
import json
import streamlit as st
import pickle
from dotenv import load_dotenv

# Funzione per caricare i dati JSON
def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

# Funzione per estrarre l'ultima parte dell'URL solo se inizia con uno dei prefissi specificati
def extract_name(url):
    prefixes = [
        "http://d3fend.mitre.org/ontologies/d3fend.owl#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://example.org/entities/",
        "http://example.org/d3f/",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    ]
    for prefix in prefixes:
        if url.startswith(prefix):
            return re.split(r'[#/]', url)[-1]
    return url  # Restituisce l'URL originale se nessun prefisso corrisponde


# Funzione per fare la similarity search
def similarity_search(question, path_similarity):
    # Carica gli embedding dal file JSON
    embeddings_data = load_embeddings(path_similarity)

    embedding_model = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model="text-embedding-ada-002",
    )
    query_vector = embedding_model.embed_query(question)

    # Calcola la similarità per ciascun embedding
    similarities = []
    for entity, embedding in embeddings_data.items():
        entity_name = entity
        embedding = np.array(embedding)
        similarity = cosine_similarity(query_vector, embedding)
        similarities.append((entity_name, similarity))


    # Ordina per similarità decrescente
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Restituisci i top_k risultati
    return similarities[:4]

# Funzione per generare una risposta dal modello LLM
def generate_RAG_answer(question: str, context: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-3.5-turbo"
    )
    prompt = f"Answer the question based on the context: \n\nContext: {context}\n\nQuestion: {question}"
    print(prompt)
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

def get_context(question, path_get_context, path_similarity):
    context = []
    results = similarity_search(question, path_similarity)
    # Carica i dati dal file pickle
    with open(path_get_context, 'rb') as f:
        train_triples, valid_triples, test_triples = pickle.load(f)

    # Combina tutte le triple
    all_triples = train_triples + valid_triples + test_triples

    # Crea una nuova lista di triple con i valori aggiornati
    processed_triples = [
        (extract_name(triple[0]), extract_name(triple[1]), extract_name(triple[2]))
        for triple in all_triples
    ]
    # Elaborazione
    for entity_name, similarity in results:
        # Rimuove gli spazi dal primo elemento della tupla
        normalized_entity_name = entity_name.replace(" ", "")
        
        # Filtra le triple che corrispondono al target_string
        filtered_triples = [
            triple for triple in processed_triples 
            if triple[0] == normalized_entity_name or triple[2] == extract_name(normalized_entity_name)
        ]
        
        # Aggiunge i risultati al contesto
        context.append((filtered_triples, similarity))

    return results,context

# Funzione per formattare i risultati
def format_similarity_results(results):
    formatted = "Similarity Search Entities:\n"
    
    for entity, similarity in results:
        formatted += f"- {entity}: {similarity:.4f}\n"
    
    return formatted

def format_triples(triples):
    formatted = "\nAssociated Triples:\n"
    
    for triple_group, similarity in triples:
        formatted += f"\nSimilarity: {similarity:.4f}\n"
        for triple in triple_group:
            formatted += f"  - {triple[0]} {triple[1]} {triple[2]}\n"
    
    return formatted


def main():
    load_dotenv(".env", override=True)
    st.title("RAG Chatbot with TransE embeddings")
    st.write("Ask your question and get answers from both RAG and LLM!")

    # Aggiungi un selectbox per scegliere il dataset
    dataset_choice = st.selectbox(
        "Choose your dataset:",
        ("d3fend", "architecture")
    )

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
            # Carica i dati in base alla selezione del dataset
            if dataset_choice == "d3fend":
                path_get_context = os.getenv('output_path')
                path_similarity = os.getenv("embeddings_file")
            else:  # 'architecture'
                path_get_context = os.getenv('output_path_arch')
                path_similarity = os.getenv("embeddings_file_arch")

            # Ottieni il contesto e le triple per la risposta
            context, triples = get_context(question, path_get_context, path_similarity)

            formatted_context = format_similarity_results(context)
            formatted_triples = format_triples(triples)
            st.session_state["context"] = formatted_context + formatted_triples

            # Genera la risposta RAG
            st.session_state["rag_answer"] = generate_RAG_answer(question, triples)

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
   
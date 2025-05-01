import os
import re
import json
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_name(url):
    prefixes = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://example.org/stix#",
        "http://d3fend.mitre.org/ontologies/d3fend.owl#",
        "http://example.org/network#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2004/02/skos/core#"
    ]
    for prefix in prefixes:
        if url.startswith(prefix):
            return re.split(r'[#/]', url)[-1]
    return url

def similarity_search(question, path_similarity):
    embeddings_data = load_embeddings(path_similarity)
    embedding_model = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model="text-embedding-ada-002",
    )
    query_vector = embedding_model.embed_query(question)
    embeddings_list = [np.array(embedding) for embedding in embeddings_data.values()]
    entity_names = list(embeddings_data.keys())
    similarities = cosine_similarity([query_vector], embeddings_list)[0]
    return sorted(zip(entity_names, similarities), key=lambda x: x[1], reverse=True)[:5]

def generate_RAG_answer(question: str, context: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-4o-mini"
    )
    prompt = [
        ("system", """
        You are an AI assistant designed to support a security analyst in monitoring, detecting, and mitigating DDoS and DoS attacks.  
        Your primary goal is to enhance the analyst's cyber situation awareness by providing concise, context-aware insights.  

        # Instructions  
        - Use only the information provided in the context. Do not use any external sources.  
        - Prioritize clear and concise answers that directly assist the analyst.  
        - Focus on practical insights that improve the analyst’s decision-making.  
        """),
        ("human", f"Context:\n{context}\n\nQuestion:\n{question}")
    ]
    return llm.invoke(prompt).content

def generate_LLM_answer(question: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-4o-mini"
    )
    return llm.invoke(question).content

def get_context(question, path_get_context, path_similarity):
    results = similarity_search(question, path_similarity)
    with open(path_get_context, 'rb') as f:
        train_triples, valid_triples, test_triples = pickle.load(f)
    all_triples = train_triples + valid_triples + test_triples
    processed_triples = [
        (extract_name(triple[0]), extract_name(triple[1]), extract_name(triple[2]))
        for triple in all_triples
    ]
    context = [
        ([triple for triple in processed_triples if extract_name(entity) in triple], similarity)
        for entity, similarity in results
    ]
    return results, context

def format_similarity_results(results):
    return "Similarity Search Entities:\n" + "\n".join(f"- {entity}: {similarity:.4f}" for entity, similarity in results)

def format_triples(triples):
    return "\nAssociated Triples:\n" + "\n".join(
        f"\nSimilarity: {similarity:.4f}\n" + "\n".join(f"  - {t[0]} {t[1]} {t[2]}" for t in triple_group)
        for triple_group, similarity in triples
    )

def rag():
    load_dotenv(".env", override=True)
    st.title("🔐 Security Analyst AI Assistant")
    st.write("Ask your cybersecurity-related questions.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if user_input := st.chat_input("Enter your question..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        path_get_context = os.getenv('output_path_arch')
        path_similarity = os.getenv("embeddings_file_arch")
        context, triples = get_context(user_input, path_get_context, path_similarity)
        formatted_context = format_similarity_results(context)
        formatted_triples = format_triples(triples)
        rag_answer = generate_RAG_answer(user_input, formatted_triples)
        llm_answer = generate_LLM_answer(user_input)
        with st.chat_message("assistant"):
            st.markdown(rag_answer)
        st.session_state["messages"].append({"role": "assistant", "content": rag_answer})
        with st.expander("🔍 Show LLM Answer"):
            st.markdown(llm_answer)
        with st.expander("📚 Show Context"):
            st.markdown(formatted_context + formatted_triples)

if __name__ == "__main__":
    rag()
import os
from langchain_openai import ChatOpenAI
import numpy as np
import re
import json
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_name(url):
    prefixes = [
        "http://d3fend.mitre.org/ontologies/d3fend.owl#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://example.org/entities/",
        "http://example.org/d3f/",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2004/02/skos/core#",
        "http://example.org/network#",
        "http://example.org/stix#",
        "http://www.w3.org/2000/01/rdf-schema#"
                ]
    for prefix in prefixes:
        if url.startswith(prefix):
            return re.split(r'[#/]', url)[-1]
    return url

# Funzione per interrogare l'LLM e ottenere la query Cypher
def search(question):
    load_dotenv(".env", override=True)
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = 'neo4j'
    
    kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
    )

    # Connessione all'API OpenAI
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-4o-mini"
    )

    # Prompt per il modello
    prompt = [
        ("system",
        """
        You are an AI that translates natural language queries into Cypher queries. 
        Your task is to output only the Cypher query with no additional text.
        Here are some examples:

        Natural language: "I detected many incoming UDP packets to my network that have 'ANY' as an argument. What might this be due to and what services in my network might be affected?"

        MATCH (dc:ns2__XMitreDataComponent)
        RETURN DISTINCT dc.uri AS uri
        UNION
        MATCH (ap:ns2__AttackPattern)
        RETURN DISTINCT ap.uri AS uri

        
        Natural language: "Are there DNS, NTP, or other UDP-based services in my network?"

        MATCH (n)
        WHERE tolower(n.rdfs__label) CONTAINS "dns" OR tolower(n.rdfs__label) CONTAINS "ntp"
        RETURN n.uri

        
        Natural language: "Can you explain how a DNS amplification attack works?"

        MATCH (n)
        WHERE tolower(n.rdfs__label) CONTAINS "amplification" 
        RETURN n.uri

        
        Natural language: "What services use DNSServer for domain name resolution in my network?"

        MATCH (n)
        WHERE tolower(n.rdfs__label) CONTAINS "dns"
        RETURN n.uri

        
        Natural language: "How can I mitigate a DNS amplification attack if the target is an SMTPServer?"

        MATCH (n)
        WHERE tolower(n.rdfs__label) CONTAINS "amplification" 
        RETURN n.uri  
        """),
        ("human", f"Question:\n{question}"),
    ]

    # Ottenere la query Cypher generata
    response = llm.invoke(prompt)
    cypher_query = response.content.strip()

    print(f"Generated Cypher Query:\n{cypher_query}")

    # Connettersi al database Neo4j e eseguire la query
    result = kg.query(cypher_query)
    
    uris = [entry['n.uri'] for entry in result]

    return uris

# Funzione per generare una risposta dal modello LLM
def generate_RAG_answer(question: str, context: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-4o-mini"
    )

    prompt = [
    (
        "system",
        """
        You are an AI assistant designed to support a security analyst in monitoring, detecting, and mitigating DDoS and DoS attacks.  
        Your primary goal is to enhance the analyst's cyber situation awareness by providing concise, context-aware insights.  

        # Instructions  
        - Use only the information provided in the context. Do not use any external sources.  
        - Prioritize clear and concise answers that directly assist the analyst.  
        - Focus on practical insights that improve the analyst’s decision-making.  
        """,
    ),
    ("human", f"Context:\n{context}\n\nQuestion:\n{question}"),
    ]

    response = llm.invoke(prompt)
    return response.content

def generate_LLM_answer(question: str):
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_TOKEN"),
        model_name="gpt-4o-mini"
    )
    response = llm.invoke(question)
    return response.content

def get_context(question, path_get_context):
    context = []
    results = search(question)
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
    for entity_name in results:        
        # Filtra le triple che corrispondono al target_string
        filtered_triples = [
            triple for triple in processed_triples 
            if triple[0] == extract_name(entity_name) or triple[2] == extract_name(entity_name) 
        ]
        
        # Aggiunge i risultati al contesto
        context.append((filtered_triples, 1.0))

    return results,context

# Funzione per formattare i risultati
def format_similarity_results(results):
    formatted = "Similarity Search Entities:\n"
    
    for entity in results:
        formatted += f"- {entity}\n"
    
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
        context, triples = get_context(user_input, path_get_context)
        formatted_context = format_similarity_results(context)
        formatted_triples = format_triples(triples)
        rag_answer = generate_RAG_answer(user_input, triples)
        llm_answer = generate_LLM_answer(user_input)
        
        with st.chat_message("assistant"):
            st.markdown(f"{rag_answer}")
        st.session_state["messages"].append({"role": "assistant", "content": rag_answer})
        
        with st.expander("🔍 Show LLM Answer"):
            st.markdown(llm_answer)
        
        with st.expander("📚 Show Context"):
            st.markdown(formatted_context + formatted_triples)

if __name__ == "__main__":
    main()

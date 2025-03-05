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

        MATCH (s:ns0__Network)-[p:ns1__contains]->(o)
        WHERE tolower(o.uri) CONTAINS "dns" OR tolower(o.uri) CONTAINS "ntp"
        RETURN o.uri AS uri

        
        Natural language: "Can you explain how a DNS amplification attack works?"

        MATCH (ap:ns2__AttackPattern)
        WHERE tolower(ap.rdfs__label) CONTAINS "amplification" 
        RETURN ap.uri AS subject, "description" AS predicate, ap.ns2__description AS object

        
        Natural language: "How can I mitigate a DNS amplification attack?"

        MATCH (s)-[p:`ns2__mitigated-by`]->(o)
        WHERE tolower(s.rdfs__label) CONTAINS "amplification"
        RETURN s.uri AS subject, type(p) AS predicate, o.uri AS object
        UNION
        MATCH (s)-[p:`ns2__mitigated-by`]->(o)
        WHERE tolower(s.rdfs__label) CONTAINS "amplification"
        RETURN o.uri AS subject, "description" AS predicate, o.ns2__description AS object
        """),
        ("human", f"Question:\n{question}"),
    ]

    # Ottenere la query Cypher generata
    response = llm.invoke(prompt)
    cypher_query = response.content.strip()
    
    # Connettersi al database Neo4j e eseguire la query
    result = kg.query(cypher_query)
    uris = []
    triples = []

    for entry in result:
        if 'subject' in entry and 'predicate' in entry and 'object' in entry:
            triples.append((extract_name(entry['subject']), extract_name(entry['predicate']), extract_name(entry['object'])))
        elif 'uri' in entry:
            uris.append(entry['uri'])

    return uris,triples, cypher_query

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
        - Answer exclusively based on the information provided in the context; do not use your own pre-existing knowledge or external sources. 
        - Prioritize clear and concise answers that directly assist the analyst.  
        - Focus on practical insights that improve the analyst’s decision-making.  
        - After 
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
    results,triples, cypher_query = search(question)
    if triples:
        for triple in triples:
            context.append((triple, 1.0))
    else:
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
    return results,context,cypher_query

# Funzione per formattare i risultati
def format_similarity_results(results):
    formatted = "Similarity Search Entities:\n"
    
    for entity in results:
        formatted += f"- {entity}\n"
    
    return formatted

def format_triples(triples, flag):
    formatted = "\nTriples:\n"
    for triple_group, similarity in triples:
        formatted += f"\nSimilarity: {similarity:.4f}\n"
        if flag == 0:
            formatted += f"- {triple_group}\n"
        else:
            for triple in triple_group:
                formatted += f"  - {triple[0]} {triple[1]} {triple[2]}\n"
    
    return formatted

def main():
    load_dotenv(".env", override=True)
    st.title("🔒 Security Analyst AI Assistant")
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
        context, triples, cypher_query = get_context(user_input, path_get_context)  # Recupera la query Cypher generata
        flag = 0
        if context:
            formatted_context = format_similarity_results(context)
            flag = 1
        else:
            formatted_context = ""
        formatted_triples = format_triples(triples, flag)
        rag_answer = generate_RAG_answer(user_input, triples)
        llm_answer = generate_LLM_answer(user_input)
        
        with st.chat_message("assistant"):
            st.markdown(f"{rag_answer}")
        st.session_state["messages"].append({"role": "assistant", "content": rag_answer})
        
        with st.expander("🔍 Show LLM Answer"):
            st.markdown(llm_answer)
        
        with st.expander("📚 Show Context"):
            st.markdown(formatted_context + formatted_triples)
        
        with st.expander("📊 Show Generated Cypher Query"):
            st.code(cypher_query, language="cypher")

if __name__ == "__main__":
    main()


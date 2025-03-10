# Harmonia: Therapy Chatbot with GraphRAG and ArangoDB
## Overview
Harmonia is a therapy-focused chatbot designed to provide support and resources for individuals dealing with mental health issues, particularly those related to violence and trauma. Built using GraphRAG, ArangoDB, and NVIDIA cuGraph, Harmonia leverages graph-based knowledge representation and natural language processing to deliver intelligent, adaptive responses. The chatbot dynamically processes user queries, retrieves relevant information from a knowledge graph, and provides empathetic, actionable insights.
## Features
- **Natural Language Query Processing:**
    - Accepts user queries in natural language.
    - Dynamically determines the intent and retrieves relevant information.
- **Graph-Based Knowledge Representation:**
    - Converts PDF content into a graph structure using NetworkX.
    - Persists and queries the graph using ArangoDB.
- **Hybrid Query Execution:**
    - Combines AQL (ArangoDB Query Language) for graph traversal and cuGraph/NetworkX for advanced analytics.
    - Handles both simple and complex queries seamlessly.
- **Therapy Resource Recommendations:**
    - Provides tailored resources and coping strategies based on user input.
    - Maintains a supportive, empathetic tone throughout interactions.
- **Visualization:**
    - Visualizes the knowledge graph and query results using matplotlib.
- **Gradio Interface:**
    - Offers an interactive, user-friendly interface for real-time interactions.
## Technologies Used
- GraphRAG: For graph-based knowledge representation and retrieval.
- ArangoDB: For graph persistence and querying.
- NVIDIA cuGraph: For GPU-accelerated graph analytics.
- LangChain: For natural language processing and agentic workflows.
- Gemini LLM: For intent recognition and response generation.
- NetworkX: For graph construction and in-memory processing.
- Gradio: For the chatbot interface.
- spaCy: For entity extraction and NLP tasks.
## Setup Instructions
- **Prerequisites**
    - Python 3.10 or higher.
    - CUDA-compatible GPU (optional, for cuGraph acceleration).
    - ArangoDB instance (local or cloud-hosted).
    - Gemini API key.
  - **Installation**
      - Clone the repository:
        ```
        git clone https://github.com/your-repo/harmonia.git
        cd harmonia```

      - Install dependencies:

       `pip install -r requirements.txt`

      - Download the spaCy model:`python -m spacy download en_core_web_sm`
      - Set up ArangoDB:
        - Create a database named TherapyChatbot.
        - Update the connection details in the code.
      - Add your Gemini API key

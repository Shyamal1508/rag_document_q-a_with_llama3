RAG Document Q&A with LLaMA 3
This project demonstrates a Retrieval-Augmented Generation (RAG) system that enables users to query documents using natural language questions. It leverages the capabilities of Meta's LLaMA 3 language model to provide accurate and context-aware answers based on the content of uploaded documents.

Features
Document Upload: Users can upload PDF documents to the system.
Contextual Question Answering: Ask questions in natural language, and the system retrieves relevant information from the uploaded documents to generate answers.
LLaMA 3 Integration: Utilizes the LLaMA 3 language model for generating human-like responses.

Usage
Prepare the Documents:
Place the PDF documents you wish to query in the project directory or specify their paths when prompted.

Run the Application:
streamlit run app.py

rag_document_q-a_with_llama3/
├── app.py                    # Main application script
├── requirements.txt          # List of required Python packages
├── ShyamalMalhotraResume.pdf # Sample PDF document
├── imagenet.pdf              # Sample PDF document
├── transformers.pdf          # Sample PDF document
└── README.md                 # Project documentation

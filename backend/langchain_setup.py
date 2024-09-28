# backend/langchain_chain.py
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import json


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def load_documents():
    # Load documents from JSON file
    with open("data/parts_data.json", "r") as f:
        data = json.load(f)

    documents = []
    for item in data:
        # Extract information from 'troubleshooting_tips'
        troubleshooting_tips = item.get("troubleshooting_tips", "")

        # Use regular expressions to parse 'troubleshooting_tips'
        symptoms_match = re.search(
            r"This part fixes the following symptoms:(.*?)This part works with the following products:",
            troubleshooting_tips,
            re.DOTALL,
        )
        symptoms = symptoms_match.group(1).strip() if symptoms_match else ""

        products_match = re.search(
            r"This part works with the following products:(.*?)Part#",
            troubleshooting_tips,
            re.DOTALL,
        )
        products = products_match.group(1).strip() if products_match else ""

        replaces_match = re.search(
            r"Part#.*?replaces these:(.*?)(?:Back to Top|$)",
            troubleshooting_tips,
            re.DOTALL,
        )
        replaces = replaces_match.group(1).strip() if replaces_match else ""

        # Combine relevant fields into a single string
        page_content = (
            f"Part Name: {item.get('part_name', '')}\n"
            f"Part Number: {item.get('part_number', '')}\n"
            f"Manufacturer: {item.get('manufacturer', '')}\n"
            f"Price: ${item.get('price', '')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Symptoms Fixed: {symptoms}\n"
            f"Works With Products: {products}\n"
            f"Replaces Part Numbers: {replaces}\n"
            f"Rating: {item.get('rating', '')}%\n"
            f"Review Count: {item.get('review_count', '')}\n"
            f"Appliance: {item.get('appliance', '')}\n"
        )
        # Create a Document with page_content and metadata
        doc = Document(
            page_content=page_content,
            metadata={
                "part_number": item.get("part_number", ""),
                "part_name": item.get("part_name", ""),
                "appliance": item.get("appliance", ""),
            },
        )
        documents.append(doc)
    return documents


def create_vectorstore(documents):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise


def save_vectorstore(vectorstore, index_path):
    vectorstore.save_local(index_path)


def load_vectorstore(index_path, embeddings):
    vectorstore = FAISS.load_local(index_path, embeddings)
    return vectorstore


def get_qa_chain():
    documents = load_documents()
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Initialize the language model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0
    )

    # Define a custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant specializing in refrigerator and dishwasher parts.
Use the following information to answer the user's question.
If the information is not sufficient, politely inform the user.

Context:
{context}

Question:
{question}

Answer:
""",
    )

    # Set up the RetrievalQA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )
    return qa_chain

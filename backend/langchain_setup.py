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
from langchain.schema import BaseRetriever
import json
from typing import Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# custom retriever class to match part names directly
class CustomRetriever(BaseRetriever):
    vectorstore: Any

    def get_relevant_documents(self, query):
        logging.info(f"Received query: {query}")
        # Check if the query contains a part number
        part_number = self.extract_numerical_identifiers(query)
        if part_number:
            logging.info(f"Extracted part number: {part_number}")
            # Search for documents with matching part number in metadata
            docs = self.search_by_part_number(part_number)
            if docs:
                logging.info(f"Found {len(docs)} document(s) matching part number.")
                return docs
            else:
                logging.info("No documents found matching part number.")
        else:
            logging.info("No part number found in query.")

        # Fallback to vector similarity search
        logging.info("Performing vector similarity search.")
        docs = self.vectorstore.similarity_search(query, k=3)
        logging.info(
            f"Retrieved {len(docs)} document(s) from vector similarity search."
        )
        return docs
        # numerical_identifiers = self.extract_numerical_identifiers(query)
        # if numerical_identifiers:
        #     # Retrieve documents based on exact metadata match
        #     docs = self.search_by_numerical_metadata(numerical_identifiers)
        #     if docs:
        #         return docs
        # # Fallback to vector similarity search
        # return self.vectorstore.similarity_search(query, k=3)

    # def extract_numerical_identifiers(self, query):
    #     # Extend to capture other numerical patterns
    #     identifiers = re.findall(r"\b(PS|WP|AP|W\d{6,7})\d*\b", query)
    #     return identifiers
    def extract_numerical_identifiers(self, query):
        # Corrected regex pattern
        identifiers = re.findall(
            r"\b(PS\d+|WP\d+|AP\d+|W\d{6,7})\b", query, re.IGNORECASE
        )
        identifiers = [id.upper() for id in identifiers]
        return identifiers[0] if identifiers else None

    def search_by_part_number(self, part_number):
        # Access the documents in the vectorstore
        all_docs = self.vectorstore.docstore._dict.values()
        # Filter documents by part number metadata
        matching_docs = [
            doc for doc in all_docs if doc.metadata.get("part_number") == part_number
        ]
        return matching_docs

    # def extract_numerical_identifiers(self, query):
    #     # Regular expressions to extract part numbers and other numerical data
    #     identifiers = re.findall(r"PS\d+|WP\d+", query)
    #     return identifiers

    def search_by_numerical_metadata(self, identifiers):
        # Access the documents in the vectorstore
        all_docs = self.vectorstore.docstore._dict.values()
        # Filter documents by numerical metadata
        matching_docs = []
        for doc in all_docs:
            metadata = doc.metadata
            # Check for exact matches in 'part_number' and 'replaces' fields
            if any(metadata.get("part_number") == id for id in identifiers):
                matching_docs.append(doc)
            elif "replaces" in metadata and any(
                id in metadata["replaces"] for id in identifiers
            ):
                matching_docs.append(doc)
        return matching_docs


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

        page_content = (
            f"The part number {item.get('part_number', '')}, which corresponds to the {item.get('part_name', '')}, is manufactured by {item.get('manufacturer', '')}.\n"
            f"This part costs ${item.get('price', '')} and is designed to {item.get('description', '')}.\n"
            f"Common symptoms indicating the need for Part Number {item.get('part_number', '')} include: {symptoms}.\n"
            f"It works with the following products: {products}.\n"
            f"Customers have rated this part {item.get('rating', '')}% with {item.get('review_count', '')} reviews.\n"
            f"Appliance Type: {item.get('appliance', '')}.\n"
        )

        # Include replaced part numbers in a sentence
        replaces = replaces.strip()
        if replaces:
            page_content += f"Part Number {item.get('part_number', '')} replaces these part numbers: {replaces}.\n"
        # Combine relevant fields into a single string
        # page_content = (
        #     f"Part Name: {item.get('part_name', '')}\n"
        #     f"Part Number: {item.get('part_number', '')}\n"
        #     f"Manufacturer: {item.get('manufacturer', '')}\n"
        #     f"Price: ${item.get('price', '')}\n"
        #     f"Description: {item.get('description', '')}\n"
        #     f"Symptoms Fixed: {symptoms}\n"
        #     f"Works With Products: {products}\n"
        #     f"Replaces Part Numbers: {replaces}\n"
        #     f"Rating: {item.get('rating', '')}%\n"
        #     f"Review Count: {item.get('review_count', '')}\n"
        #     f"Appliance: {item.get('appliance', '')}\n"
        # )
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
    # retriever = vectorstore.as_retriever(
    #     search_type="similarity", search_kwargs={"k": 3}
    # )
    retriever = CustomRetriever(vectorstore=vectorstore)

    # Initialize the language model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0
    )

    # Define a custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant specializing in refrigerator and dishwasher parts.
Pay closse attention to part numbers, model numbers, and other identifiers. You will be given the part number corresponding with each part and should be able to provide the corresponding name if given a part number, along with related details.
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

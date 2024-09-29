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
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
import json
from typing import Any, List
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

    def extract_numerical_identifiers(self, query):
        # an attempt to get a correct regex pattern
        identifiers = re.findall(
            r"\b(PS\d+|WP\d+|AP\d+|W\d{6,7})\b", query, re.IGNORECASE
        )
        identifiers = [id.upper() for id in identifiers]
        return identifiers[0] if identifiers else None

    def search_by_part_number(self, part_number):
        all_docs = self.vectorstore.docstore._dict.values()
        # Filter documents by part number metadata
        matching_docs = [
            doc for doc in all_docs if doc.metadata.get("part_number") == part_number
        ]
        return matching_docs

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
    with open("data/parts_data.json", "r") as f:
        data = json.load(f)

    documents = []
    for item in data:
        troubleshooting_tips = item.get("troubleshooting_tips", "")

        # Assuming standardized formatting of the troubleshooting section, which appears to be the case
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
            f"It is compatible with the following models: {' '.join(item.get('compatible_models', ''))}"
        )

        replaces = replaces.strip()
        if replaces:
            page_content += f"Part Number {item.get('part_number', '')} replaces these part numbers: {replaces}.\n"
        doc = Document(
            page_content=page_content,
            metadata={
                "part_number": item.get("part_number", ""),
                "part_name": item.get("part_name", ""),
                "appliance": item.get("appliance", ""),
                "compatible_models": [
                    model.upper() for model in item.get("compatible_models", [])
                ],
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


def get_question_rephrase_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Creates an LLMChain to rephrase follow-up questions into standalone questions.
    """
    condense_question_prompt = PromptTemplate.from_template("""
    Given the following conversation and a follow-up question, rephrase the question to be a standalone question.

    Conversation:
    {chat_history}

    Follow-up question:
    {question}

    Standalone question:
    """)
    condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
    return condense_question_chain


def get_answer_generation_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Creates an LLMChain to generate answers based on retrieved context and chat history.
    """
    qa_prompt = PromptTemplate.from_template("""
    You are a helpful assistant specializing in refrigerator and dishwasher parts.
    Pay close attention to part numbers, model numbers, and other identifiers.
    Use the following conversation history and context to answer the user's question.
    If the information is not sufficient, politely inform the user.
    Do not make up any information that you do not have.
    If you are asked anything that does not directly relate to refrigerator parts or dishwasher parts or refrigerators or dishwashers, or does not directly reference something that is in the context provided below, politely inform the user that you are only able to provide information about refrigerator and dishwasher parts.

    Conversation History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)
    answer_chain = LLMChain(llm=llm, prompt=qa_prompt)
    return answer_chain


def process_query(
    user_query: str,
    chat_history: List[dict],
    retriever: CustomRetriever,
    rephrase_chain: LLMChain,
    answer_chain: LLMChain,
) -> str:
    """
    Processes the user's query by rephrasing it, retrieving relevant documents,
    and generating an answer.
    """
    # Format chat history into a string
    formatted_chat_history = ""
    for message in chat_history:
        role = message.get("role", "")
        content = message.get("content", "")
        formatted_chat_history += f"{role.capitalize()}: {content}\n"

    standalone_question = rephrase_chain.run(
        chat_history=formatted_chat_history, question=user_query
    )

    logging.info(f"Standalone question: {standalone_question}")

    retrieved_docs = retriever.get_relevant_documents(standalone_question)

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    logging.info(f"Retrieved context: {context}")

    answer = answer_chain.run(
        chat_history=formatted_chat_history, context=context, question=user_query
    )

    logging.info(f"Assistant response: {answer}")

    return answer


def get_qa_chain():
    documents = load_documents()
    logging.info(f"Loaded {len(documents)} documents.")

    vectorstore = create_vectorstore(documents)
    logging.info("Vector store created.")

    retriever = CustomRetriever(vectorstore=vectorstore)

    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0
    )

    # Initialize the LLMChains
    rephrase_chain = get_question_rephrase_chain(llm)
    answer_chain = get_answer_generation_chain(llm)

    logging.info("LLM chains initialized.")

    return {
        "rephrase_chain": rephrase_chain,
        "answer_chain": answer_chain,
        "retriever": retriever,
    }

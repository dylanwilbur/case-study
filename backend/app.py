from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from langchain_setup import get_qa_chain, process_query
from pydantic import BaseModel
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    chat_history: list = []


# Multi-chained architecture to handle chat history
qa_components = get_qa_chain()
rephrase_chain = qa_components["rephrase_chain"]
answer_chain = qa_components["answer_chain"]
retriever = qa_components["retriever"]


@app.get("/")
async def hello_world():
    return "Hello world"


@app.post("/getAIMessage")
async def get_ai_message(request: QueryRequest):
    try:
        user_query = request.query
        chat_history = request.chat_history

        assistant_response = process_query(
            user_query=user_query,
            chat_history=chat_history,
            retriever=retriever,
            rephrase_chain=rephrase_chain,
            answer_chain=answer_chain,
        )

        response = {
            "assistant_response": assistant_response,
            "chat_history": chat_history,  # Return the updated chat history
        }

        return response

    except Exception as e:
        print(f"Error handling /getAIMessage: {e}")
        response = {
            "message": {
                "role": "assistant",
                "content": "An error occurred on the server.",
            }
        }
        return JSONResponse(content=response, status_code=500)

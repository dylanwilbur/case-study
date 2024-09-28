from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from langchain_setup import get_qa_chain
from pydantic import BaseModel
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )  # Use openai directly, not OpenAI class

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


# Initialize the QA chain once when the app starts
qa_chain = get_qa_chain()


@app.get("/")
async def hello_world():
    return "Hello World from the Python backend!"


@app.post("/getAIMessage")
async def get_ai_message(request: QueryRequest):
    try:
        user_query = request.query
        print(f"Received query: {user_query}")

        assistant_response = qa_chain.run(user_query)
        # assistant_response = qa_chain({"query": user_query})
        response = {"message": {"role": "assistant", "content": assistant_response}}
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

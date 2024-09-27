from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  # Use openai directly, not OpenAI class

load_dotenv()


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello_world():
    return "Hello World from the Python backend!"


@app.post("/getAIMessage")
async def get_ai_message(request: Request):
    try:
        data = await request.json()
        user_query = data.get("query", "")

        # Use OpenAI's ChatCompletion API
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in refrigerator and dishwasher parts.",
                },
                {"role": "user", "content": user_query},
            ],
        )

        assistant_response = completion.choices[0].message.content
        response = {"message": {"role": "assistant", "content": assistant_response}}
        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error handling /getAIMessage: {e}")
        response = {
            "message": {
                "role": "assistant",
                "content": "An error occurred on the server.",
            }
        }
        return JSONResponse(content=response, status_code=500)

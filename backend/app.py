from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello World from the Python backend!"


@app.route("/getAIMessage", methods=["POST"])
def get_ai_message():
    try:
        data = request.get_json()
        user_query = data.get("query", "")

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in refrigerator and dishwasher parts.",
                },
                {"role": "user", "content": user_query},
            ],
        )

        assistant_response = completion.choices[0].message["content"]
        response = {"message": {"role": "assistant", "content": assistant_response}}
        return jsonify(response)
    except Exception as e:
        # For now, just respond with a simple message
        print(f"Error handling /getAIMessage: {e}")
        response = {
            "message": {
                "role": "assistant",
                "content": "An error occurred on the server.",
            }
        }
        return jsonify(response), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(port=port, debug=True)

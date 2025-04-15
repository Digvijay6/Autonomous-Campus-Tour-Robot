import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPEN_AI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPEN_AI_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=AZURE_OPEN_AI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT
)

SYSTEM_PROMPT = """
You are a friendly campus tour guide assistant for Graphic Era University. 
Provide concise, engaging information about the query.
Keep responses under 100 words and maintain a cheerful tone.
"""


def get_campus_info(query):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        print("OpenAI response:", reply)
        return reply
    except Exception as e:
        print("Azure OpenAI SDK Error:", e)
        return None
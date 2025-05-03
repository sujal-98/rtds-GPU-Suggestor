import os
import json
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Load FAISS index
faiss_index_path = "gpu_instances.index"
index = faiss.read_index(faiss_index_path)

# Load metadata list
with open("gpu_instances_metadata.json", "r") as f:
    id_to_metadata = json.load(f)
metadata_list = list(id_to_metadata.values())

# Gemini prompt processor
def gpt_process(prompt: str) -> dict:
    system_instruction = (
        "You are a helpful assistant that extracts GPU instance configuration from prompts. "
        "Return a JSON object with these fields: vcpus, ram, price_per_hour, "
        "price_per_month, price_per_spot, is_gpu, is_spot, is_public. Only return the JSON."
    )

    try:
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
        response = model.generate_content([
            {"role": "user", "parts": [system_instruction + "\n\n" + prompt]}
        ])

        text_response = response.text.strip()
        print(text_response)
        # Try to locate JSON block if the model wraps it in explanation
        json_str = text_response
        if "{" in text_response:
            json_str = text_response[text_response.find("{"):text_response.rfind("}") + 1]

        return json.loads(json_str)

    except Exception as e:
        print("Error:", e)
        return {
            "vcpus": 16,
            "ram": 32,
            "price_per_hour": 0.85,
            "price_per_month": 600,
            "price_per_spot": 0.3,
            "is_gpu": 1,
            "is_spot": 0,
            "is_public": 1
        }

# Request schema
class InputQuery(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the GPU Resource Search API (Gemini version)!"}

@app.post("/search")
def search_resources(query: InputQuery):
    print("User prompt:", query.prompt)

    gpt_output = gpt_process(query.prompt)

    vector = np.array([
        gpt_output['vcpus'],
        gpt_output['ram'],
        gpt_output['price_per_hour'],
        gpt_output['price_per_month'],
        gpt_output['price_per_spot'],
        gpt_output['is_gpu'],
        gpt_output['is_spot'],
        gpt_output['is_public']
    ], dtype='float32').reshape(1, -1)

    k = 5
    distances, indices = index.search(vector, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < len(metadata_list):
            results.append({
                "distance": float(distances[0][i]),
                "match": metadata_list[idx]
            })

    return {"results": results}

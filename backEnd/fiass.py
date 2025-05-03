import faiss
import numpy as np
import requests
import hashlib
import json
import os

# Initialize Faiss index
dim = 9
index = faiss.IndexFlatL2(dim)

# API URLs
api_urls = [
    "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region=ap-south-mum-1",
    "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region=ap-south-noi-1",
    "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region=us-east-at-1"
]

# Fetch and extract "data" from API response
def fetch_data_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

# Convert item to 8D vector
def convert_to_vector(item):
    return np.array([
        item['vcpus'],
        item['ram'],
        item['price_per_hour'],
        item['price_per_month'],
        item['price_per_spot'],
        item['is_gpu'],
        item['is_spot'],
        item['is_public'],

    ], dtype="float32")

# Generate unique hash for metadata
def generate_id(item):
    key_fields = {
        "resource_name": item["resource_name"],
        "region": item["region"],
        "ram": item["ram"],
        "vcpus": item["vcpus"]
    }
    key_str = json.dumps(key_fields, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

# Fetch and process all data
all_data = []
vectors = []
id_to_metadata = {}

for url in api_urls:
    data = fetch_data_from_api(url)
    for item in data:
        try:
            vector = convert_to_vector(item)
            vectors.append(vector)

            item_id = generate_id(item)
            id_to_metadata[item_id] = item
        except KeyError as e:
            print(f"Skipping due to missing key: {e}")

# Add to FAISS index
vectors_np = np.array(vectors)
index.add(vectors_np)

# Save FAISS index
faiss.write_index(index, "gpu_instances.index")

# Save mapping (index to metadata)
with open("gpu_instances_metadata.json", "w") as f:
    json.dump(id_to_metadata, f)

print(f"✅ Total vectors stored: {index.ntotal}")
print(f"✅ Total metadata records stored: {len(id_to_metadata)}")

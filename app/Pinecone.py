import os
from pinecone import Pinecone
import time

PINECONE_API_KEY= "pcsk_6g4mfa_2XpjjMk75wd1ZdrKS3cq7JaMGg1Z5ntD25BfivdXnJdpX6xLk6hhdCojqKR7cvt"

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "nmoot"

# Delete index if it exists and wait for deletion to complete
if index_name in pc.list_indexes():
    pc.delete_index(index_name)
    print(f"Waiting for index '{index_name}' deletion...")
    while True:
        time.sleep(5)
        if index_name not in pc.list_indexes():
            print("Index deletion confirmed")
            break

# Remove the extra sleep and add a check before creation
if index_name not in pc.list_indexes():
    dimension = 384
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index '{index_name}' already exists")

data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

print(index.describe_index_stats())

query = "Tell me about the tech company known as Apple."

embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

results = index.query(
    namespace="ns1",
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(f"Results: {results}")
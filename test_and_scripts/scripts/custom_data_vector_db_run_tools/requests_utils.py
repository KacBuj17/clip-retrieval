import yaml
from PIL import Image
import time
from image_embedding_utils import ImageEmbedder
from text_embedding_utils import TextEmbedder


def run_qdrant_request(qdrant_url, qdrant_api, collection_name, vector_name, limit, query):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api)

    query = query.squeeze()
    query = query.tolist()

    start = time.time()
    response = client.query_points(
        collection_name=collection_name,
        query=query,
        using=vector_name,
        limit=limit,
        with_payload=True
    )
    stop = time.time()

    response_time = stop - start
    print(f"response time: {response_time}s")

    results = []
    for point in response.points:
        results.append({
            "id": point.id,
            "payload": point.payload,
            "vector": point.vector,
            "score": getattr(point, "score", None)
        })

    return results


def run_mongo_request(mongo_uri, mongo_db_name, collection_name, vector_name, limit, query):
    return


def run_db_request(query, vector_name, db_type, limit):
    with open("../configs/db_config.yaml", "r") as f:
        db_config = yaml.safe_load(f)

    if db_type == "mongo":
        mongo_uri = db_config["mongo_uri"]
        mongo_db_name = db_config["mongo_db_name"]
        collection_name = db_config["collection_name"]
        return run_mongo_request(mongo_uri, mongo_db_name, collection_name, vector_name, limit, query)

    if db_type == "qdrant":
        qdrant_url = db_config["qdrant_url"]
        qdrant_api = db_config["qdrant_api"]
        collection_name = db_config["collection_name"]
        return run_qdrant_request(qdrant_url, qdrant_api, collection_name, vector_name, limit, query)


def run_request(input_data, input_type="text", vector_name="text", db_type="qdrant", limit=10, use_mclip=False):
    if input_type == "text":
        embedder = TextEmbedder(use_mclip=use_mclip)
        vec = embedder.encode(input_data)
    elif input_type == "image":
        image = input_data
        if isinstance(input_data, str):
            image = Image.open(input_data).convert("RGB")
        embedder = ImageEmbedder()
        vec = embedder.encode(image)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    return run_db_request(query=vec, vector_name=vector_name, db_type=db_type, limit=limit)


def main():
    input_type = input("Enter input type(text or image): ")
    if input_type != "text" and input_type != "image":
        raise ValueError(f"Unsupported input_type: {input_type}")

    vector_name = input("Enter request type(text or image): ")
    if vector_name != "text" and vector_name != "image":
        raise ValueError(f"Unsupported request_type: {vector_name}")

    if input_type == "text":
        input_data = input("Enter text: ")
    elif input_type == "image":
        input_data = input("Enter image path: ")
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    try:
        limit = int(input("Enter results limit(int > 0): "))
    except ValueError:
        print("[ValueError]: Please enter a positive integer")
        return

    results = run_request(input_data=input_data, input_type=input_type, vector_name=vector_name, limit=limit,
                          use_mclip=False)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()

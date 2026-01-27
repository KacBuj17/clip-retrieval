import requests

def query_text(text, url, limit=5, vector_name="image"):
    payload = {
        "type": "text",
        "data": text,
        "vector_name": vector_name,
        "limit": limit
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error {response.status_code}: {response.text}")


def query_image(image_path, url, limit=5, vector_name="image"):
    payload = {
        "type": "image",
        "data": image_path,
        "vector_name": vector_name,
        "limit": limit
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    print(query_text(text="cat", limit=1, vector_name="image", url=f"http://127.0.0.1:5000/query"))
    print(query_image(image_path="example.jpg", limit=1, vector_name="image", url=f"http://127.0.0.1:5000/query"))

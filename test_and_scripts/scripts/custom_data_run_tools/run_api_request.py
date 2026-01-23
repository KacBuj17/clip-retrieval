from clip_retrieval.clip_client import ClipClient
import os

def send_request(url, indice_name, text, num_images):
    client = ClipClient(url=url, indice_name=indice_name, num_images=num_images)
    results = client.query(text=text)
    return results

def main():
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    url = f"http://{backend_host}:{backend_port}/knn-service"

    indice_name = os.environ.get("INDICE_NAME", "example_index")
    text = os.environ.get("TEXT", "example text")
    num_images = int(os.environ.get("NUM_IMAGES", 5))

    results = send_request(url, indice_name, text, num_images)
    print(results[0])

if __name__ == "__main__":
    main()
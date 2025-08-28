from clip_retrieval.clip_client import ClipClient

def send_request(url, indice_name, text):
    client = ClipClient(url=url, indice_name=indice_name)
    results = client.query(text=text)
    return results

def main():
    url = 'http://127.0.0.1:1234/knn-service'
    indice_name = 'example_index'
    text = 'example text'

    results = send_request(url, indice_name, text)
    print(results[0])

if __name__ == "__main__":
    main()
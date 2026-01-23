from PIL import Image
from flask import Flask, request, jsonify
from requests_utils import run_db_request, TextEmbedder, ImageEmbedder

def run_backend(host="0.0.0.0", port=5000, use_mclip=False, db_type="qdrant"):
    app = Flask(__name__)

    text_embedder = TextEmbedder(use_mclip=use_mclip)
    image_embedder = ImageEmbedder()

    def encode_input(input_data, input_type="text"):
        if input_type == "text":
            return text_embedder.encode(input_data)
        elif input_type == "image":
            image = input_data
            if isinstance(input_data, str):
                image = Image.open(input_data).convert("RGB")
            return image_embedder.encode(image)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    @app.route("/query", methods=["POST"])
    def query_endpoint():
        data = request.get_json()
        if not data or "type" not in data or "data" not in data:
            return jsonify({"error": "Missing 'type' or 'data' in request"}), 400

        input_type = data["type"]
        input_data = data["data"]
        vector_name = data.get("vector_name", input_type)
        limit = data.get("limit", 10)

        try:
            limit = int(limit)
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            vec = encode_input(input_data, input_type=input_type)
            results = run_db_request(query=vec, vector_name=vector_name, db_type=db_type, limit=limit)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({"results": results})

    app.run(host=host, port=port)

if __name__ == "__main__":
    run_backend(host="0.0.0.0", port=5000, use_mclip=False, db_type="qdrant")

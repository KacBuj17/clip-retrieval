import json
import math
import uuid
from io import BytesIO

import fsspec
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


# ====================== MONGO ======================
class MongoOutputSink:
    """Mongo sink with batching"""

    def __init__(self, mongo_uri, db_name, collection_name,
                 enable_text=True, enable_image=True, enable_metadata=True,
                 batch_size=1024):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.batch_size = batch_size

        self.documents = []
        self.client = None
        self.db = None
        self.collection = None

    def _init_client(self):
        if self.client is None:
            self.client = MongoClient(self.mongo_uri, server_api=ServerApi('1'), connect=False)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            try:
                self.client.admin.command('ping')
                print(f"[Mongo:{self.collection_name}] Connected")
            except Exception as e:
                print(f"[Mongo:{self.collection_name}] Connection error:", e)

    def add(self, sample):
        self._init_client()
        new_docs = []

        if sample.get("image_embs") is not None:
            n = sample["image_embs"].shape[0]
        elif sample.get("text_embs") is not None:
            n = sample["text_embs"].shape[0]
        else:
            return

        for i in range(n):
            sample_id = str(uuid.uuid4())

            doc = {
                "_id": sample_id,
                "image_embedding": None,
                "text_embedding": None,
                "image_path": None,
                "caption": None,
                "metadata": {}
            }

            if self.enable_image and sample.get("image_embs") is not None:
                doc["image_embedding"] = sample["image_embs"][i].tolist()
                doc["image_path"] = sample["image_filename"][i]

            if self.enable_text and sample.get("text_embs") is not None:
                doc["text_embedding"] = sample["text_embs"][i].tolist()
                doc["caption"] = sample["text"][i]

            if self.enable_metadata and sample.get("metadata") is not None:
                meta = sample["metadata"][i]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {"raw_metadata": meta}
                elif not isinstance(meta, dict):
                    meta = dict(meta)
                doc["metadata"] = meta

            new_docs.append(doc)

        self.documents.extend(new_docs)

        if len(self.documents) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.documents:
            return
        self._init_client()
        try:
            self.collection.insert_many(self.documents)
            print(f"[Mongo:{self.collection_name}] Inserted {len(self.documents)} documents")
        except Exception as e:
            print(f"[Mongo:{self.collection_name}] Error inserting documents:", e)
        self.documents = []


class MongoWriter:
    """Writer class for MongoDB in buffer style"""

    def __init__(self, mongo_uri, db_name, collection_name,
                 enable_text=True, enable_image=True, enable_metadata=True):
        self.sink = MongoOutputSink(mongo_uri, db_name, collection_name,
                                    enable_text, enable_image, enable_metadata)

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()


# ====================== QDRANT ======================
class QdrantOutputSink:
    """Qdrant sink with batching"""

    def __init__(self, url, api_key, collection_name="default",
                 enable_text=True, enable_image=True, enable_metadata=True,
                 vector_size=512, batch_size=1024):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.vector_size = vector_size
        self.batch_size = batch_size

        self.points = []
        self.client = None

    def _init_client(self):
        if self.client is None:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in existing:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "image": {"size": self.vector_size, "distance": "Cosine"},
                        "text": {"size": self.vector_size, "distance": "Cosine"}
                    }
                )
            print(f"[Qdrant:{self.collection_name}] Connected to Qdrant Cloud")

    def add(self, sample):
        self._init_client()
        new_points = []

        if sample.get("image_embs") is not None:
            n = sample["image_embs"].shape[0]
        elif sample.get("text_embs") is not None:
            n = sample["text_embs"].shape[0]
        else:
            return

        for i in range(n):
            sample_id = str(uuid.uuid4())

            vectors = {}
            payload = {}

            if self.enable_image and sample.get("image_embs") is not None:
                vectors["image"] = sample["image_embs"][i].tolist()
                payload["image_path"] = sample["image_filename"][i]

            if self.enable_text and sample.get("text_embs") is not None:
                vectors["text"] = sample["text_embs"][i].tolist()
                payload["caption"] = sample["text"][i]

            if self.enable_metadata and sample.get("metadata") is not None:
                meta = sample["metadata"][i]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {"raw_metadata": meta}
                elif not isinstance(meta, dict):
                    meta = dict(meta)
                payload.update(meta)

            new_points.append(
                PointStruct(
                    id=sample_id,
                    vector=vectors,
                    payload=payload
                )
            )

        self.points.extend(new_points)

        if len(self.points) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.points:
            return
        self._init_client()
        try:
            self.client.upsert(collection_name=self.collection_name, points=self.points)
            print(f"[Qdrant:{self.collection_name}] Inserted {len(self.points)} points")
        except Exception as e:
            print(f"[Qdrant:{self.collection_name}] Error inserting points:", e)
        self.points = []


class QdrantWriter:
    """Qdrant writer in buffer style"""

    def __init__(self, url, api_key, collection_name="default",
                 enable_text=True, enable_image=True, enable_metadata=True, vector_size=512):
        self.sink = QdrantOutputSink(url, api_key, collection_name,
                                     enable_text, enable_image, enable_metadata,
                                     vector_size)

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()


class NumpyOutputSink:
    """This output sink can save image, text embeddings as npy and metadata as parquet"""

    def __init__(self, output_folder, enable_text, enable_image, enable_metadata, partition_id, output_partition_count):
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.fs, output_folder = fsspec.core.url_to_fs(output_folder)
        self.output_folder = output_folder
        self.img_emb_folder = output_folder + "/img_emb"
        self.text_emb_folder = output_folder + "/text_emb"
        self.metadata_folder = output_folder + "/metadata"
        self.batch_num = partition_id
        self.oom_partition_count = int(math.log10(output_partition_count)) + 1

        if enable_image:
            self.fs.makedirs(self.img_emb_folder, exist_ok=True)

        if enable_text:
            self.fs.makedirs(self.text_emb_folder, exist_ok=True)

        self.fs.makedirs(self.metadata_folder, exist_ok=True)

        self.batch_count = 0
        self.__init_batch()

    def __init_batch(self):
        self.image_embeddings = []
        self.text_embeddings = []
        self.image_names = []
        self.captions = []
        self.metadata = []
        self.batch_count = 0

    def add(self, sample):
        """
        add to buffers the image embeddings, text embeddings, and meta
        """

        self.batch_count += sample["image_embs"].shape[0] if self.enable_image else sample["text_embs"].shape[0]
        if self.enable_image:
            self.image_embeddings.append(sample["image_embs"])
            self.image_names.extend(sample["image_filename"])
        if self.enable_text:
            self.captions.extend(sample["text"])
            self.text_embeddings.append(sample["text_embs"])
        if self.enable_metadata:
            self.metadata.extend(sample["metadata"])

    def __write_batch(self):
        """
        write a batch of embeddings and meta to npy and parquet
        """
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pandas as pd  # pylint: disable=import-outside-toplevel

        data_lists = []
        data_columns = []
        batch_num_str = str(self.batch_num).zfill(self.oom_partition_count)
        if self.enable_image:
            img_emb_mat = np.concatenate(self.image_embeddings)
            output_path_img = self.img_emb_folder + "/img_emb_" + batch_num_str

            with self.fs.open(output_path_img + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, img_emb_mat)
                f.write(npb.getbuffer())

            data_lists.append(self.image_names)
            data_columns.append("image_path")

        if self.enable_text:
            text_emb_mat = np.concatenate(self.text_embeddings)
            output_path_text = self.text_emb_folder + "/text_emb_" + batch_num_str

            with self.fs.open(output_path_text + ".npy", "wb") as f:
                npb = BytesIO()
                np.save(npb, text_emb_mat)
                f.write(npb.getbuffer())

            data_lists.append(self.captions)
            data_columns.append("caption")

        if self.enable_metadata:
            data_lists.append(self.metadata)
            data_columns.append("metadata")

        df = pd.DataFrame(data=list(zip(*data_lists)), columns=data_columns)
        if self.enable_metadata:
            parsed_metadata = pd.json_normalize(df["metadata"].apply(json.loads))
            without_existing_columns = parsed_metadata.drop(
                columns=set(["caption", "metadata", "image_path"]) & set(parsed_metadata.keys())
            )
            df = df.join(without_existing_columns).drop(columns=["metadata"])

        output_path_metadata = self.metadata_folder + "/metadata_" + batch_num_str + ".parquet"
        with self.fs.open(output_path_metadata, "wb") as f:
            df.to_parquet(f)

    def flush(self):
        if self.batch_count == 0:
            return
        self.__write_batch()
        self.__init_batch()


class NumpyWriter:
    """the numpy writer writes embeddings to folders img_emb, text_emb, and metadata"""

    def __init__(self, partition_id, output_folder, enable_text, enable_image, enable_metadata, output_partition_count):
        self.sink = NumpyOutputSink(
            output_folder, enable_text, enable_image, enable_metadata, partition_id, output_partition_count
        )

    def __call__(self, batch):
        self.sink.add(batch)

    def flush(self):
        self.sink.flush()

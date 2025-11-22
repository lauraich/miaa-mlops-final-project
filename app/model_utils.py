import os
import json
import numpy as np
import onnxruntime as ort
from datetime import datetime

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class ModelManager:
    def __init__(self, storage_account, container, model_blob, log_blob, conn_string):
        """
        storage_account: nombre de la cuenta de storage: ej. miaamlopsresources
        container: nombre del contenedor donde está el modelo y logs
        model_blob: ruta del archivo ONNX dentro del container
        log_blob: archivo .txt donde quedarán predicciones
        conn_string: connection string de Azure Storage
        """
        self.storage_account = storage_account
        self.container = container
        self.model_blob = model_blob
        self.log_blob = log_blob
        self.conn_string = conn_string

        self.blob_service = BlobServiceClient.from_connection_string(conn_string)
        self.container_client = self.blob_service.get_container_client(container)

        self.local_model_path = os.path.join("/tmp", os.path.basename(model_blob))
        self.session = None

    # ------------------------------
    # Descarga del modelo
    # ------------------------------
    def ensure_model(self):
        if not os.path.exists(self.local_model_path):
            self.download_model()
        if self.session is None:
            self.session = ort.InferenceSession(self.local_model_path)

    def download_model(self):
        blob = self.container_client.get_blob_client(self.model_blob)
        with open(self.local_model_path, "wb") as file:
            file.write(blob.download_blob().readall())

    # ------------------------------
    # Predicción ONNX
    # ------------------------------
    def predict(self, inputs):
        arr = np.array(inputs, dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: arr})
        return [o.tolist() if hasattr(o, "tolist") else o for o in output]

    # ------------------------------
    # Logging de predicciones
    # ------------------------------
    def log_prediction(self, prediction):
        log_blob_client = self.container_client.get_blob_client(self.log_blob)

        # Intentar leer el archivo actual
        try:
            existing = log_blob_client.download_blob().readall().decode("utf-8")
        except Exception:
            existing = ""

        now = datetime.utcnow().isoformat()
        new_line = json.dumps({"timestamp": now, "prediction": prediction}) + "\n"
        updated = existing + new_line

        # Sobrescribir blob
        log_blob_client.upload_blob(updated, overwrite=True)

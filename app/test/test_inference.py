from azure.storage.blob import BlobClient
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app
from model_utils import ModelManager
import tempfile

TEST_IMAGES_CONTAINER = os.getenv("TEST_IMAGES_CONTAINER")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

def test_download_test_image():
    conn = AZURE_STORAGE_CONNECTION_STRING
    container = TEST_IMAGES_CONTAINER
    blob_name = "personas.jpg"

    blob = BlobClient.from_connection_string(conn, container, blob_name)

    data = blob.download_blob().readall()

    assert len(data) > 0

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None


def test_full_prediction_pipeline():
    mgr = ModelManager(
        storage_account=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        container=os.getenv("AZURE_CONTAINER_NAME"),
        log_container=os.getenv("AZURE_LOG_CONTAINER_NAME"),
        model_blob=os.getenv("AZURE_MODEL_BLOB"),
        log_blob=os.getenv("AZURE_LOG_BLOB_NAME"),
        conn_string=AZURE_STORAGE_CONNECTION_STRING
    )

    mgr.ensure_model()

    blob = BlobClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING,
        TEST_IMAGES_CONTAINER,
        "personas.jpg"
    )
    img_bytes = blob.download_blob().readall()

    # Preprocesar como en el API
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)

    detections = mgr.predict(img)

    assert isinstance(detections, list)
    assert len(detections) > 0


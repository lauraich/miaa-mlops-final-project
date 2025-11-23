from azure.storage.blob import BlobClient
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app
from model_utils import ModelManager

def test_download_test_image():
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("TEST_IMAGES_CONTAINER")
    blob_name = "test1.jpg"

    blob = BlobClient.from_connection_string(conn, container, blob_name)

    data = blob.download_blob().readall()

    assert len(data) > 0

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None


def test_full_prediction_pipeline():
    mgr = ModelManager(
        storage_account="miaamlopsresources",
        container=os.getenv("AZURE_CONTAINER_NAME"),
        model_blob=os.getenv("AZURE_MODEL_BLOB"),
        log_blob="test_predictions.txt",
        conn_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )

    # Ensure the model exists
    mgr.ensure_model()

    # Download test image
    blob = BlobClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        os.getenv("TEST_IMAGES_CONTAINER"),
        "test_person.jpg"
    )
    img_bytes = blob.download_blob().readall()

    # Preprocesar como en tu API
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300, 300))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)

    detections = mgr.predict(img)

    assert isinstance(detections, list)
    assert len(detections) > 0


client = TestClient(app)

def test_predict_endpoint():
    img_path = "tests/test_person.jpg"

    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": f})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000  # debería tener bytes
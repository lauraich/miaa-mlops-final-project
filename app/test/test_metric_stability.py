import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from model_utils import ModelManager
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from azure.storage.blob import BlobClient
import cv2
import numpy as np
load_dotenv()

ENV = os.getenv("ENVIRONMENT", "dev")
AZ_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME")
MODEL_BLOB = os.getenv("AZURE_MODEL_BLOB")
LOG_BLOB = "predicciones_dev.txt" if ENV == "dev" else "predicciones_prod.txt"
STORAGE_ACCOUNT = "miaamlopsresources"

model = ModelManager(
    storage_account="miaamlopsresources",
    container=os.getenv("AZURE_CONTAINER_NAME"),
    model_blob=os.getenv("AZURE_MODEL_BLOB"),
    log_container=os.getenv("AZURE_LOG_CONTAINER_NAME"),
    log_blob="test_predictions.txt",
    conn_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)


model.ensure_model()

# Download test image
blob = BlobClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    os.getenv("TEST_IMAGES_CONTAINER"),
    "personas.jpg"
)
img_bytes = blob.download_blob().readall()

# Preprocesar como en tu API
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
img = cv2.resize(img, (300, 300))
img = img.astype(np.uint8)
img = np.expand_dims(img, axis=0)

detections = model.predict(img)
y_test = [
    {
        'box': {'top': 0.284269779920578, 'left': 0.6359336972236633, 'bottom': 0.9263092279434204, 'right': 0.8839356303215027}, 
        'class_index': 1, 
        'score': 0.8715940713882446
    }, 
    {
        'box': {'top': 0.1876823455095291, 'left': 0.08408525586128235, 'bottom': 0.642768144607544, 'right': 0.2473745048046112}, 
        'class_index': 1, 
        'score': 0.8348629474639893
    }
]


def test_model_loads():
    assert model is not None


def test_detection_count_stability():
    assert len(detections) == len(y_test)

def detection_accuracy(y_test, y_pred):
    y_true_classes = [o["class_index"] for o in y_test]
    y_pred_classes = [o["class_index"] for o in y_pred]

    correct = sum(1 for t, p in zip(y_true_classes, y_pred_classes) if t == p)
    return correct / len(y_test)

def test_min_accuracy():
    acc = detection_accuracy(y_test, detections)
    assert acc >= 0.80

def test_class_stability():
    pred_classes = [d["class_index"] for d in detections]
    true_classes = [d["class_index"] for d in y_test]

    assert pred_classes == true_classes


def test_score_not_significantly_lower():
    pred_scores = [d["score"] for d in detections]
    true_scores = [d["score"] for d in y_test]

    for p, t in zip(pred_scores, true_scores):
        assert p >= t - 0.10  # tolerancia de caída de 10%


def test_model_drift_limited():
    print("Detections:", detections)
    print("Y Test:", y_test)
    historical_score = 0.85
    current_score = np.mean([d["score"] for d in detections])
    print(f"current_score: {current_score}, historical_score: {historical_score}")

    assert abs(historical_score - current_score) <= 0.05
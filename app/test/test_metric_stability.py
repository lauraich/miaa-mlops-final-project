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

model = ModelManager(
    storage_account=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
    container=os.getenv("AZURE_CONTAINER_NAME"),
    log_container=os.getenv("AZURE_LOG_CONTAINER_NAME"),
    model_blob=os.getenv("AZURE_MODEL_BLOB"),
    log_blob=os.getenv("AZURE_LOG_BLOB_NAME"),
    conn_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)

model.ensure_model()

# Download test image
blob = BlobClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    os.getenv("TEST_IMAGES_CONTAINER"),
    "cat.jpeg"
)
img_bytes = blob.download_blob().readall()

# Preprocesar como en tu API
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_h, input_w = 300, 300
img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
img = img.astype(np.uint8)
img = np.expand_dims(img, axis=0)

detections = model.predict(img)
y_test = [
    {'box': {'bottom': 0.7944386005401611, 'left': 0.37576597929000854, 'right': 0.6311351656913757, 'top': 0.2239646017551422}, 'class_index': 17, 'score': 0.9384123086929321}
]


def test_model_loads():
    assert model is not None


def test_detection_count_stability():
    print(f"model : {model}\n")
    print(f"Detections : {detections}\n")
    print(f"Y Test : {y_test}\n")
    
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
    historical_score = 0.936
    current_score = np.mean([d["score"] for d in detections])

    assert abs(historical_score - current_score) <= 0.05
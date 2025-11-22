import os
import json
import numpy as np
from model_utils import ModelManager
import boto3

def load_test_data_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj['Body'].read()
    return json.loads(body)

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def test_metric_not_worse_than_threshold():
    bucket = os.getenv("TEST_DATA_BUCKET")
    test_key = os.getenv("TEST_DATA_KEY")  # ej: test/data.json
    model_bucket = os.getenv("MODEL_BUCKET")
    model_key = os.getenv("MODEL_KEY")
    data = load_test_data_from_s3(bucket, test_key)
    X = data['X']  # lista
    y_true = data['y']
    mgr = ModelManager(bucket=model_bucket, model_key=model_key, log_key="test_preds.txt")
    mgr.ensure_model()
    preds = []
    for x in X:
        preds.append(mgr.predict(x)[0])  # ajustar deserialización según salida
    mae = compute_mae(y_true, preds)
    # Umbral (ejemplo): MAE <= 0.2
    assert mae <= 0.2, f"MAE too high: {mae}"

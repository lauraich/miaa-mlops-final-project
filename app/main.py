from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from model_utils import ModelManager

app = FastAPI()

ENV = os.getenv("ENVIRONMENT", "dev")
AZ_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME")          # ej: models
MODEL_BLOB = os.getenv("AZURE_MODEL_BLOB")                # ej: model/my_model.onnx
LOG_BLOB = "predicciones_dev.txt" if ENV == "dev" else "predicciones_prod.txt"
STORAGE_ACCOUNT = "miaamlopsresources"

model_manager = ModelManager(
    storage_account=STORAGE_ACCOUNT,
    container=AZ_CONTAINER,
    model_blob=MODEL_BLOB,
    log_blob=LOG_BLOB,
    conn_string=AZ_CONN_STR,
)

class InputPayload(BaseModel):
    inputs: list

@app.on_event("startup")
def startup_event():
    model_manager.ensure_model()

@app.post("/predict")
def predict(payload: InputPayload):
    try:
        out = model_manager.predict(payload.inputs)
        model_manager.log_prediction(out)
        return {"prediction": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

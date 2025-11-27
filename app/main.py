from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from model_utils import ModelManager
import uvicorn
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENVIRONMENT", "dev") 
AZ_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("AZURE_CONTAINER_NAME")
AZ_LOG_CONTAINER = os.getenv("AZURE_LOG_CONTAINER_NAME")
MODEL_BLOB = os.getenv("AZURE_MODEL_BLOB")
LOG_BLOB = os.getenv("AZURE_LOG_BLOB_NAME")
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

model_manager = ModelManager(
    storage_account=STORAGE_ACCOUNT,
    container=AZ_CONTAINER,
    log_container=AZ_LOG_CONTAINER,
    model_blob=MODEL_BLOB,
    log_blob=LOG_BLOB,
    conn_string=AZ_CONN_STR,
)


def preprocess_image_bytes(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_h, input_w = 300, 300
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    return img


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.ensure_model()
    print(f"Modelo cargado para entorno: {ENV} ✔️")
    yield
    print("Cerrando aplicación...")


app = FastAPI(lifespan=lifespan)


# Static
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>No se encontró index.html</h1>", status_code=404)
    return FileResponse(index_path)


# Cambia el endpoint según la rama
ENDPOINT_PREFIX = "/predict" if ENV == "dev" else f"/predict-{ENV}"
print(f"Usando endpoint: {ENDPOINT_PREFIX} for entorno: {ENV}")

@app.post(ENDPOINT_PREFIX)
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        #print("Bytes recibidos:", len(contents))

        img_array = preprocess_image_bytes(contents)
        #print("Imagen preprocesada correctamente:", img_array.shape)

        detections = model_manager.predict(img_array)
        #print("Predicción generada:", detections)

        model_manager.log_prediction(detections)
        #print("Predicción registrada.")

        img_with_boxes = model_manager.draw_detections(contents, detections)
        #print("Imagen con cajas generada.")

        return HTMLResponse(
            content=img_with_boxes,
            media_type="image/jpeg"
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error interno: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

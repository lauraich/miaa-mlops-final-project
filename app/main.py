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
LOG_BLOB = "predicciones_dev.txt" if ENV == "dev" else "predicciones_prod.txt"
STORAGE_ACCOUNT = "miaamlopsresources"

model_manager = ModelManager(
    storage_account=STORAGE_ACCOUNT,
    container=AZ_CONTAINER,
    log_container=AZ_LOG_CONTAINER,
    model_blob=MODEL_BLOB,
    log_blob=LOG_BLOB,
    conn_string=AZ_CONN_STR,
)

# Modificamos el pre-procesamiento para aceptar bytes directos
def preprocess_image_bytes(img_bytes: bytes):
    # Decodificar array de bytes directamente a imagen OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen. Asegúrate de enviar un formato válido (jpg, png).")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    
    img = img.astype(np.uint8) # Aseguramos que sea uint8
    img = np.expand_dims(img, axis=0) # (1, 300, 300, 3)
    return img

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.ensure_model()
    print("Modelo cargado ✔️")
    yield
    print("Cerrando aplicación...")

app = FastAPI(lifespan=lifespan)

#app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = "app/static/index.html"
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>No se encontró index.html</h1>", status_code=404)
    return FileResponse(index_path)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        # Leer los bytes del archivo directamente
        contents = await file.read()

        img_array = preprocess_image_bytes(contents)
        print("Imagen preprocesada para predicción.")
        detections = model_manager.predict(img_array)
        model_manager.log_prediction(detections)

        # Dibujar cajas sobre la imagen original
        img_with_boxes = model_manager.draw_detections(contents, detections)

        return HTMLResponse(
            content=img_with_boxes,
            media_type="image/jpeg"
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error interno: {str(e)}") # Log en consola para debug
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
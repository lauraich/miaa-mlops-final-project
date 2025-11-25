import os
import tempfile
import json
import numpy as np
import onnxruntime as ort
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
import cv2

from labels import COCO_LABELS


class ModelManager:
    def __init__(self, storage_account, container,log_container, model_blob, log_blob, conn_string, score_threshold=0.5):
        """
        storage_account: nombre de la cuenta de storage: ej. miaamlopsresources
        container: nombre del contenedor donde está el modelo y logs
        model_blob: ruta del archivo ONNX dentro del container
        log_blob: archivo .txt donde quedarán predicciones
        conn_string: connection string de Azure Storage
        score_threshold: umbral de confianza para filtrar predicciones (default 0.5)
        """
        self.storage_account = storage_account
        self.container = container
        self.log_container = log_container
        self.model_blob = model_blob
        self.log_blob = log_blob
        self.conn_string = conn_string
        self.score_threshold = score_threshold

        self.blob_service = BlobServiceClient.from_connection_string(conn_string)
        self.container_client = self.blob_service.get_container_client(container)
        self.log_container_client = self.blob_service.get_container_client(log_container)

        # Usar /tmp es más seguro en entornos cloud/containers
        self.local_model_path = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(model_blob)
        )
        self.session = None

    # ------------------------------
    # Descarga del modelo
    # ------------------------------
    def ensure_model(self):
        print("Local model path:", self.local_model_path)

        # Si existe pero está vacío, eliminarlo
        if os.path.exists(self.local_model_path) and os.path.getsize(self.local_model_path) == 0:
            print("Archivo ONNX vacío detectado. Eliminando...")
            os.remove(self.local_model_path)

        if not os.path.exists(self.local_model_path):
            print("Modelo no existe. Descargando...")
            self.download_model()
        else:
            print("Modelo ya existe localmente.")

        print("Tamaño final del archivo:", os.path.getsize(self.local_model_path))

        if self.session is None:
            print("Creando sesión ONNX...")
            self.session = ort.InferenceSession(self.local_model_path)

    def download_model(self):
        blob = self.container_client.get_blob_client(self.model_blob)
        print("Blob:", blob)
        with open(self.local_model_path, "wb") as file:
            file.write(blob.download_blob().readall())

    # ------------------------------
    # Predicción ONNX
    # ------------------------------
    def predict(self, image_array: np.ndarray):
        if self.session is None:
            self.ensure_model()

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image_array})
        
        # Según la documentación y estándar TF Object Detection API exportado a ONNX:
        # outputs[0] = detection_boxes: [batch, num_detections, 4] -> (top, left, bottom, right)
        # outputs[1] = detection_classes: [batch, num_detections] -> COCO class index
        # outputs[2] = detection_scores: [batch, num_detections] -> Confidence (0-1)
        # outputs[3] = num_detections: [batch] -> Cantidad de detecciones válidas
        
        boxes = outputs[0]
        classes = outputs[1]
        scores = outputs[2]
        num_detections_array = outputs[3]
        
        detections = []
        
        # Obtenemos el número de detecciones para la primera imagen del batch (batch index 0)
        num_detections = int(num_detections_array[0]) 
        
        for i in range(num_detections):
            score = float(scores[0][i])
            
            if score > self.score_threshold:
                # box format documentation: [top, left, bottom, right] (relative)
                raw_box = boxes[0][i].tolist()
                
                # Convertir a int el índice de clase (vienen como float en algunos modelos)
                class_index = int(classes[0][i])
                
                detections.append({
                    "box": {
                        "top": raw_box[0],
                        "left": raw_box[1],
                        "bottom": raw_box[2],
                        "right": raw_box[3]
                    },
                    "class_index": class_index,
                    "score": score
                })
                
        return detections

    # ------------------------------
    # Logging de predicciones
    # ------------------------------
    def log_prediction(self, prediction):
        log_blob_client = self.log_container_client.get_blob_client(self.log_blob)

        # Intentar leer el archivo actual
        try:
            existing = log_blob_client.download_blob().readall().decode("utf-8")
        except Exception:
            existing = ""

        now = datetime.now(timezone.utc).isoformat()
        # Guardamos solo si hay predicciones, o un registro vacío indicando "no detection"
        payload = {
            "timestamp": now,
            "prediction": prediction
        }
        new_line = json.dumps(payload) + "\n"
        updated = existing + new_line

        # Sobrescribir blob
        log_blob_client.upload_blob(updated, overwrite=True)


    def draw_detections(self, original_img_bytes, detections):
        # Leer bytes → imagen BGR
        nparr = np.frombuffer(original_img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]

        for det in detections:
            box = det["box"]
            score = det["score"]

            # Las coordenadas vienen normalizadas (0–1), convertirlas
            top = int(box["top"] * h)
            left = int(box["left"] * w)
            bottom = int(box["bottom"] * h)
            right = int(box["right"] * w)

            # Dibujar rectángulo
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # Etiqueta con el score
            class_name = COCO_LABELS.get(det["class_index"], "Unknown")
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img, label, (left, top - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Codificar imagen a JPEG para enviar en respuesta
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()
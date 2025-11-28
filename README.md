# Object Detection Service · MLOps con Azure y GitHub Actions

Este proyecto implementa un servicio de detección de objetos haciendo uso del modelo **ssd_mobilenet_v1_13-qdq** utilizando **FastAPI**, empaquetado en Docker y desplegado mediante un pipeline de **CI/CD con GitHub Actions**.

En este caso la rama `main` representa el entorno de producción y `dev` representa el entorno de desarrollo.
## 🚀 Arquitectura General

El flujo de trabajo automatizado (CI/CD) se ejecuta en cada push a las ramas `dev` o `main`(prod):

1.  **GitHub Actions**: Orquesta el pipeline.
2.  **Build**: Construye la imagen Docker del servicio.
3.  **Push**: Sube la imagen a **Azure Container Registry (ACR)**.
4.  **Deploy**: Despliega la nueva versión en **Azure Container Apps**.
5.  **Model Management**: Gestiona el modelo y logs en **Azure Blob Storage**.

## 📦 Componentes del Proyecto

### Estructura de Archivos

```
.
├── .github/workflows/ci-cd.yml  # Pipeline de CI/CD
├── Dockerfile                   # Definición de la imagen Docker
├── app/
│   ├── main.py                  # Punto de entrada de la aplicación FastAPI
│   ├── model_utils.py           # Lógica de inferencia y gestión del modelo
│   ├── labels.py                # Etiquetas de clases del modelo
│   ├── requirements.txt         # Dependencias de Python
│   └── static/                  # Archivos estáticos (Frontend básico)
│   └── test/                    # Archivos con las pruebas de conexión y estabilidad del modelo
└── README.md                    # Documentación del proyecto
```

### Tecnologías Clave

*   **Python 3.11**
*   **FastAPI**: Framework web moderno y rápido.
*   **OpenCV & ONNX Runtime**: Procesamiento de imágenes e inferencia de modelos.
*   **Azure**:
    *   Container Registry (ACR)
    *   Blob Storage
    *   Container Apps
*   **Docker**: Contenerización.

## 🛠️ Configuración Local

Para ejecutar el proyecto en la máquina local:

1.  **Clonar el repositorio**:
    ```bash
    git clone <url-del-repo>
    cd miaa-mlops-final-project
    ```

2.  **Crear un entorno virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r app/requirements.txt
    ```

4.  **Configurar variables de entorno**:
    Crea un archivo `.env` en la carpeta `app/` con las siguientes variables (necesitarás las credenciales de Azure):
    ```env
    ENVIRONMENT=dev
    AZURE_STORAGE_CONNECTION_STRING=<tu_connection_string>
    AZURE_CONTAINER_NAME=<nombre_contenedor_blob>
    AZURE_LOG_CONTAINER_NAME=<nombre_contenedor_logs>
    AZURE_MODEL_BLOB=<nombre_archivo_modelo>
    AZURE_LOG_BLOB_NAME=<nombre_archivo_log>
    AZURE_STORAGE_ACCOUNT_NAME=<nombre_cuenta_storage>
    ```

5.  **Ejecutar la aplicación**:
    ```bash
    uvicorn app.main:app --reload
    ```
    La API estará disponible en `http://127.0.0.1:8000`.

## 🐳 Ejecución con Docker

Puedes construir y ejecutar el contenedor localmente para simular el entorno de producción.

1.  **Construir la imagen**:
    ```bash
    docker build -t object-identifier .
    ```

2.  **Ejecutar el contenedor**:
    Asegúrate de tener el archivo `.env` configurado.
    ```bash
    docker run -p 8080:8080 --env-file app/.env object-identifier
    ```
    La aplicación se ejecutará en `http://localhost:8080`.

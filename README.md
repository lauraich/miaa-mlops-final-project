# Servicio Médico · MLOps con Azure y GitHub Actions

Este proyecto implementa un servicio de predicción médica empaquetado en Docker y desplegado mediante un pipeline de **CI/CD con GitHub Actions**, utilizando:

- **Azure Container Registry (ACR)** para almacenar imágenes Docker  
- **Azure Blob Storage** para almacenar artefactos y modelos  
- **GitHub Actions** para automatizar build, push y despliegue  
- **Python + Flask/FastAPI** (o el framework que uses)  
- **Modelo de Machine Learning**: almacenado como `modelo.pkl`  

---

## 🚀 Arquitectura General


El pipeline se ejecuta cada vez que haces un push a `main`.

---

## 📦 Componentes del Proyecto

### 1. **Aplicación**
Código fuente del servicio médico:
- `/app/`
- `/src/`
- `/model/`

Incluye el endpoint de predicción y carga del modelo desde Azure Blob Storage.

### 2. **Dockerfile**
Define cómo se construye la imagen para producción.

### 3. **GitHub Actions Workflow**
Ubicado en:


Este workflow:
1. Compila la imagen Docker  
2. Autentica en Azure  
3. Envía la imagen al ACR  
4. Sube el modelo a Blob Storage  
5. (Opcional) Despliega la app en Azure  

---

## 🗂️ Estructura del Repositorio


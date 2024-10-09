# dockerfiles/streamlit/Dockerfile
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar las dependencias de Python desde el archivo requirements.txt
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Configurar las variables de entorno para las credenciales y el endpoint de MinIO
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV AWS_ENDPOINT_URL_S3=http://s3:9000
ENV MLFLOW_S3_ENDPOINT_URL=http://s3:9000

# Copiar el código de la aplicación Streamlit al contenedor
# COPY ../../streamlit /app

# Exponer el puerto de Streamlit
EXPOSE 8501

# Verificación de la salud del contenedor
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio para ejecutar la aplicación de Streamlit
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

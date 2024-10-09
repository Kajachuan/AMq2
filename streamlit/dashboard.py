import streamlit as st
import pandas as pd
import awswrangler as wr
import boto3
import matplotlib.pyplot as plt

# Configurar la sesión de boto3 para conectarse a MinIO
session = boto3.Session(
    aws_access_key_id='minio',
    aws_secret_access_key='minio123'
)

# Configurar el cliente S3 para MinIO usando el nombre del servicio de MinIO en la red de Docker
s3_client = session.client(
    service_name='s3',
    endpoint_url='http://s3:9000'  # Usar el nombre del servicio 's3'
)

# Leer el archivo CSV desde MinIO usando awswrangler
bucket_name = "data"
file_path = "raw/weatherAUS.csv"

try:
    # Utilizar awswrangler para leer el archivo CSV desde MinIO
    data = wr.s3.read_csv(path=f's3://{bucket_name}/{file_path}', boto3_session=session)
except Exception as e:
    st.error(f"Error al conectar con MinIO o al leer el archivo: {e}")

# Verificar si se cargaron los datos correctamente
if 'data' in locals() and data is not None:
    # Convertir la columna 'Date' a tipo datetime y extraer el mes y año
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Month'] = data['Date'].dt.to_period('M')

    # Contar la cantidad de observaciones por mes
    observations_per_month = data['Month'].value_counts().sort_index()

    # Graficar la cantidad de observaciones por mes
    st.title('Cantidad de Observaciones por Mes')
    st.bar_chart(observations_per_month)

    st.write("Datos cargados correctamente y visualización generada.")
else:
    st.warning("No se pudieron cargar los datos. Verifica la conexión y la ruta del archivo.")

import streamlit as st
import pandas as pd
import awswrangler as wr
import boto3
import matplotlib.pyplot as plt
import os
import requests

# FastApi
endpoint = "http://fastapi:8800/predict"

# Configurar la sesión de boto3 para conectarse a MinIO
session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
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


st.write("# Trabajo práctico final de la materia Aprendizaje de maquina 2")
st.write("##### *Esta pagina fue realizada con Streamlit y en ella podrá interactuar con el modelo de ML para predecir lluvias basado en el dataset Rain in Australia*")

st.image("rain_aust.png", caption="")

tab1, tab2, tab3, tab4 = st.tabs(["Datasets", "Gráficos", "Predicción", "Encuesta"])


#-------------------------------- PESTAÑA 1 - DATASET -------------------------------
with tab1:

    st.header("Descripcion del dataset")
    st.write("Datos:", data.describe())
    st.divider()
    #--------------------------------

    st.header("Dataset - Primeros 50 datos")
    st.write("Datos:", data.head(50) )

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(data)

    st.download_button(
        label="Descargar dataset completo como CSV",
        data=csv,
        file_name="Dataset_RIA.csv",
        mime="text/csv",
    )
    st.divider()
    #--------------------------------

    st.header("Tipos de datos del dataset")
    st.write("Datos:", data.dtypes)
    st.divider()
    #--------------------------------


#-------------------------------- PESTAÑA 2 - GRAFICOS -------------------------------
with tab2:

    st.header("Vista de Graficos")

    with st.container():
    # Verificar si se cargaron los datos correctamente
        if 'data' in locals() and data is not None:
            # Convertir la columna 'Date' a tipo datetime y extraer el mes y año
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Month'] = data['Date'].dt.to_period('M').astype(str)

            # Contar la cantidad de observaciones por mes
            observations_per_month = data['Month'].value_counts().sort_index()

            st.title('Rain in Australia')
            # Graficar la cantidad de observaciones por mes
            st.write('### Cantidad de Observaciones por Mes')
            st.bar_chart(observations_per_month)

            st.write("Datos cargados correctamente y visualización generada.")
        else:
            st.warning("No se pudieron cargar los datos. Verifica la conexión y la ruta del archivo.")

#-------------------------------- PESTAÑA 3 - PREDICCION -------------------------------
with tab3:

    st.header("Ingrese los datos y luego oprima el boton Predecir")

    fecha = st.date_input("Fecha a predecir", value=None)

    unique_locations = data['Location'].unique()
    Location = st.selectbox('Seleccione una ubicación', unique_locations)
    
    MinTemp = st.number_input("MinTemp", min_value=None, max_value=None, value=13.6)
    MaxTemp = st.number_input("MaxTemp", min_value=None, max_value=None, value=28.3)
    Rainfall = st.number_input("Rainfall", min_value=None, max_value=None, value=2.6)
    Evaporation = st.number_input("Evaporation", min_value=None, max_value=None, value=3.5)
    Sunshine = st.number_input("Temperatura minima", min_value=None, max_value=None, value=5.2)

    dir = ["E","ENE","ESE","N","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"]
    WindGustDir = st.selectbox('WindGustDir ', dir)

    WindGustSpeed = st.number_input("WindGustSpeed", min_value=None, max_value=None, value=44.3)

    WindDir9am_dir = ["E","ENE","ESE","N","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"] 
    WindDir9am = st.selectbox('WindDir9am', dir)

    WindDir3pm_dir = ["E","ENE","ESE","N","NE","NNE","NNW","NW","S","SE","SSE","SSW","SW","W","WNW","WSW"] 
    WindDir3pm = st.selectbox('WindDir3pm', dir)

    WindSpeed9am = st.number_input("WindSpeed9am", min_value=None, max_value=None, value=41.9)
    WindSpeed3pm = st.number_input("WindSpeed3pm", min_value=None, max_value=None, value=43.5)
    Humidity9am = st.number_input("Humidity9am", min_value=None, max_value=None, value=68.6)
    Humidity3pm = st.number_input("Humidity3pm", min_value=None, max_value=None, value=81.3)
    Pressure9am = st.number_input("Pressure9am", min_value=None, max_value=None, value=1008)
    Pressure3pm = st.number_input("Pressure3pm", min_value=None, max_value=None, value=1007.6)
    Cloud9am = st.number_input("Cloud9am ", min_value=None, max_value=None, value=6)
    Cloud3pm = st.number_input("Cloud3pm ", min_value=None, max_value=None, value=8)
    Temp9am = st.number_input("Temp9am", min_value=None, max_value=None, value=16.7)
    Temp3pm = st.number_input("Temp3pm ", min_value=None, max_value=None, value=25.6)

    RainToday  = st.checkbox("RainToday")

   
    if st.button("Predecir"):
        processing = st.empty()
        processing.write("Procesando petición...." ) 

        data = {
            "features": {
                "Date": str(fecha),
                "Location": Location,
                "MinTemp": MinTemp ,
                "MaxTemp": MaxTemp,
                "Rainfall": Rainfall,
                "Evaporation": Evaporation,
                "Sunshine": Sunshine,
                "WindGustDir": WindGustDir,
                "WindGustSpeed": WindGustSpeed,
                "WindDir9am": WindDir9am,
                "WindDir3pm": WindDir3pm,
                "WindSpeed9am": WindSpeed9am ,
                "WindSpeed3pm": WindSpeed3pm,
                "Humidity9am": Humidity9am ,
                "Humidity3pm": Humidity3pm ,
                "Pressure9am": Pressure9am,
                "Pressure3pm": Pressure3pm,
                "Cloud9am": Cloud9am,
                "Cloud3pm": Cloud3pm,
                "Temp9am": Temp9am,
                "Temp3pm": Temp3pm,
                "RainToday": RainToday 
            }
        }

        response = requests.post(endpoint, json=data)

        respuesta = response.json()

        processing.empty()

        if respuesta['int_output']:
            st.image("lluvia.png", caption="")
            st.write("### Anda buscando el paraguas maestro!")
        else:
            st.image("sol.png", caption="")
            st.write("### Dale tranquilo con el baile al aire libre, va estar más seco que lengua de loro")

        st.write("Respuesta completa: ", response.json())
        st.write("Código devuelto: ", response.status_code, "razon :", response.reason)

#-------------------------------- PESTAÑA 4 -------------------------------
with tab4:

    with st.form("Encuesta"):
        st.write("##### *Emita su opiñon sobre el TP final de materia*")
        checkbox_val1 = st.checkbox("Fantastico")
        checkbox_val2 = st.checkbox("Inmejorable")
        checkbox_val3 = st.checkbox("Tienen un 10+ felicitado")

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Gracias por participar")

import streamlit as st

st.write("# Bienvenido a la página del TP final de Aprendizaje de máquina 2! 👋")

st.markdown(
    """
    #### Esta página fue generada utilizado Streamlit.
    #### Aquí podrá ver datos del dataset *Rain in Australia*  y hacer predicciones según el modelo entrenado en el TP de AMq1
    
    ### Integrantes
    - Kevin Cajachuán Arroyo
    - Daniel Herrera
    - Augusto Doffo
    - Matías Marando 
    - Omar Lopez Cabrera

    ### Pasos para probar el proyecto
    - Clonar el [repositorio de GitHub](https://github.com/Kajachuan/AMq2), branch main
    - Ejecutar el archivo *docker-compose.yaml*
    - En *Airflow* verán dos DAGS.
        * etl_process
        * retrain_the_model
     - Ejecutar el DAG *etl_process*, de esta manera se crearán los datos en el bucket `s3://data`.
     - Una vez finalizado, ejecutar la notebook *experiment_mlflow.ipynb*
     - Opcionalmente, ejecutar el DAG *retrain_the_model*
     - Verificar en MLFlow la creación del experimento, y del modelo registrado
     - Verificar los datos generados en MinIO
     - Para probar el modelo:
        * Ejecutar la notebook *fastapi.ipynb*
        * Ingresar al Web Server de [Streamlit](http://localhost:8501/Dashboard)
        * En la página *Dashboard*, en la pestaña *Predicciones*, podrá indresar los datos y hacer una predicción de lluvia  
"""
)

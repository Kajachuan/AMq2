import json
import pickle
import boto3
import mlflow
import datetime

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "rain_in_australia_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the rain in Australia prediction model.

    This class defines the input fields required by the rain in Australia prediction model along with their descriptions
    and validation constraints.

    :param Date: The date of observation
    :param Location: The common name of the location of the weather station
    :param MinTemp: The minimum temperature in degrees celsius
    :param MaxTemp: The maximum temperature in degrees celsius
    :param Rainfall: The amount of rainfall recorded for the day in mm
    :param Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
    :param Sunshine: The number of hours of bright sunshine in the day.
    :param WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
    :param WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
    :param WindDir9am: Direction of the wind at 9am
    :param WindDir3pm: Direction of the wind at 3pm
    :param WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
    :param WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3pm
    :param Humidity9am: Humidity (percent) at 9am
    :param Humidity3pm: Humidity (percent) at 3pm
    :param Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
    :param Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
    :param Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
    :param Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
    :param Temp9am: Temperature (degrees C) at 9am
    :param Temp3pm: Temperature (degrees C) at 3pm
    :param RainToday: Boolean: True if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise False
    """

    Date: datetime.date = Field(
        description="The date of observation",
    )
    Location: str = Field(
        description="The common name of the location of the weather station",
    )
    MinTemp: float = Field(
        description="The minimum temperature in degrees celsius",
        ge=-10,
        le=55,
    )
    MaxTemp: float = Field(
        description="The maximum temperature in degrees celsius",
        ge=-10,
        le=55,
    )
    Rainfall: float = Field(
        description="The amount of rainfall recorded for the day in mm",
        ge=0,
        le=400,
    )
    Evaporation: float = Field(
        description="The so-called Class A pan evaporation (mm) in the 24 hours to 9am",
        ge=0,
        le=200,
    )
    Sunshine: float = Field(
        description="The number of hours of bright sunshine in the day.",
        ge=0,
        le=24,
    )
    WindGustDir: str = Field(
        description="The direction of the strongest wind gust in the 24 hours to midnight",
    )
    WindGustSpeed: float = Field(
        description="The speed (km/h) of the strongest wind gust in the 24 hours to midnight",
        ge=0,
        le=200,
    )
    WindDir9am: str = Field(
        description="Direction of the wind at 9am",
    )
    WindDir3pm: str = Field(
        description="Direction of the wind at 3pm",
    )
    WindSpeed9am: float = Field(
        description="Wind speed (km/hr) averaged over 10 minutes prior to 9am",
        ge=0,
        le=200,
    )
    WindSpeed3pm: float = Field(
        description="Wind speed (km/hr) averaged over 10 minutes prior to 3pm",
        ge=0,
        le=200,
    )
    Humidity9am: float = Field(
        description="Humidity (percent) at 9am",
        ge=0,
        le=100,
    )
    Humidity3pm: float = Field(
        description="Humidity (percent) at 3pm",
        ge=0,
        le=100,
    )
    Pressure9am: float = Field(
        description="Atmospheric pressure (hpa) reduced to mean sea level at 9am",
        ge=850,
        le=2000,
    )
    Pressure3pm: float = Field(
        description="Atmospheric pressure (hpa) reduced to mean sea level at 3pm",
        ge=850,
        le=2000,
    )
    Cloud9am: int = Field(
        description="Fraction of sky obscured by cloud at 9am. This is measured in \"oktas\", " + 
        "which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. " + 
        "A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.",
        ge=0,
        le=9,
    )
    Cloud3pm: int = Field(
        description="Fraction of sky obscured by cloud (in \"oktas\": eighths) at 3pm." + 
        "See Cload9am for a description of the values",
        ge=0,
        le=9,
    )
    Temp9am: float = Field(
        description="Temperature (degrees C) at 9am",
        ge=-10,
        le=55,
    )
    Temp3pm: float = Field(
        description="Temperature (degrees C) at 3pm",
        ge=-10,
        le=55,
    )
    RainToday: bool = Field(
        description="Boolean: True if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise False",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Date": "2024-01-01",
                    "Location": "Albury",
                    "MinTemp": 13.6,
                    "MaxTemp": 28.3,
                    "Rainfall": 2.6,
                    "Evaporation": 3.5,
                    "Sunshine": 5.2,
                    "WindGustDir": "WNW",
                    "WindGustSpeed": 44.3,
                    "WindDir9am": "WNW",
                    "WindDir3pm": "W",
                    "WindSpeed9am": 41.9,
                    "WindSpeed3pm": 43.5,
                    "Humidity9am": 68.6,
                    "Humidity3pm": 81.3,
                    "Pressure9am": 1008,
                    "Pressure3pm": 1007.6,
                    "Cloud9am": 6,
                    "Cloud3pm": 8,
                    "Temp9am": 16.7,
                    "Temp3pm": 25.6,
                    "RainToday": False
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the rain in Australia prediction model.

    This class defines the output fields returned by the rain in Australia prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if it will rain tomorrow.
    :param str_output: Output of the model in string form. Can be "It will rain tomorrow" or "It won't rain tomorrow".
    """

    int_output: bool = Field(
        description="Output of the model. True if it will rain tomorrow",
    )
    str_output: Literal["It will rain tomorrow", "It won't rain tomorrow"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "It will rain tomorrow",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("rain_in_australia_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Rain in Australia predictor API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Rain in Australia predictor API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting rain in australia.

    This endpoint receives features related to an Australian city weather and predicts whether it will rain on next day
    or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Define get season from date
    def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return 'Summer'
            elif month in [3, 4, 5]:
                return 'Fall'
            elif month in [6, 7, 8]:
                return 'Winter'
            elif month in [9, 10, 11]:
                return 'Spring'

    # Get data from date and encoding season
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    features_df['Year'] = features_df['Date'].dt.year
    features_df['Month'] = features_df['Date'].dt.month
    features_df['Day'] = features_df['Date'].dt.day
    features_df['Season'] = features_df['Date'].apply(get_season)
    features_df.drop(columns='Date', inplace=True)

    features_df['SeasonDegree'] = features_df['Season'].map(data_dict['season_degrees'])
    features_df['Season_sin'] = np.sin(np.deg2rad(features_df['SeasonDegree']))
    features_df['Season_cos'] = np.cos(np.deg2rad(features_df['SeasonDegree']))
    features_df.drop(columns=["Season"], inplace=True)
    features_df.drop(columns=["SeasonDegree"], inplace=True)

    # Encode wind dir features
    for dir_var in data_dict['wind_dir_columns']:
        features_df[dir_var] = features_df[dir_var].map(data_dict['wind_dir_degrees'])
        features_df[f'{dir_var}_sin'] = np.sin(np.deg2rad(features_df[dir_var]))
        features_df[f'{dir_var}_cos'] = np.cos(np.deg2rad(features_df[dir_var]))
        features_df.drop(columns=[dir_var], inplace=True)

    # Get latitude and longitude from location
    features_df[['Latitude', 'Longitude']] = features_df['Location'].apply(lambda x: pd.Series(data_dict['city_coordinates'][x]))
    features_df.drop(columns=['Location'], inplace=True)

    # Reorder DataFrame columns
    features_df = features_df[data_dict["columns_after_transform"]]

    # Set correct dtypes
    features_df = features_df.astype(data_dict["columns_dtypes_after_transform"])

    # Scale the data using standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "It won't rain tomorrow"
    if prediction[0] > 0:
        str_pred = "It will rain tomorrow"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)

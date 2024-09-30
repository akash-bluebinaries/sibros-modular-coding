import os
from datetime import datetime


# artifact -> pipeline folder -> timestamp -> output

def get_current_timestamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"


CURRENT_TIMESTAMP = get_current_timestamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "Data"
DATA_DIR_KEY = "labelledData.csv"

ARTIFACT_DIR_KEY = "Artifact"


# Data Ingestion related variables

DATA_INGESTION_KEY = "ingestion"
DATA_INGESTION_RAW_DATA_DIR = "rawData"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingestedData"
RAW_DATA_DIR_KEY = "rawData.csv"
TRAIN_DATA_DIR_KEY = "train_data.csv"
TEST_DATA_DIR_KEY = "test_data.csv"



# Data Transformation related variables

DATA_TRANSFORMATION_ARTIFACT = "transformation"
DATA_PREPROCESSED_DIR = "processor"
DATA_TRANSFORMATION_PROCESSING_OBJ = "processor.pkl"
DATA_TRANSFORM_DIR = "transformation"
X_TRANSFORM_TRAIN_DIR_KEY = "X_train.csv"
Y_TRANSFORM_TRAIN_DIR_KEY = "y_train.csv"
X_TRANSFORM_TEST_DIR_KEY = "X_test.csv"
Y_TRANSFORM_TEST_DIR_KEY = "y_test.csv"

# artifact/ data_transformation/ processor->processor.pkl & transformation-> train.csv & test.csv


# Model Training

MODEL_TRAINER_DIR = "model_trainer"
MODEL_OBJ = "model.pkl"


# Batch Prediction

BATCH_PREDICTION_DIR = "batch_prediction"
BATCH_PREDICTION_DATA_DIR_KEY = "data"
BATCH_PREDICTION_OUTPUT_DIR_KEY = "predictions"
BATCH_PREDICTION_OUTPUT_FILE_KEY = "predictions.csv"

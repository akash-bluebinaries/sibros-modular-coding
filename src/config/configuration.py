
import os
from src.config.constant import *



ROOT_DIR = ROOT_DIR_KEY

# DATASET_URL = 

DATASET_PATH = os.path.join(ROOT_DIR,
                            DATA_DIR,
                            DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(ROOT_DIR,
                             ARTIFACT_DIR_KEY,
                             DATA_INGESTION_KEY,
                             CURRENT_TIMESTAMP,
                             DATA_INGESTION_RAW_DATA_DIR,
                             RAW_DATA_DIR_KEY)
                             
TRAIN_FILE_PATH = os.path.join(ROOT_DIR,
                               ARTIFACT_DIR_KEY,
                               DATA_INGESTION_KEY,
                               CURRENT_TIMESTAMP,
                               DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                               TRAIN_DATA_DIR_KEY)
                        
TEST_FILE_PATH = os.path.join(ROOT_DIR,
                              ARTIFACT_DIR_KEY,
                              DATA_INGESTION_KEY,
                              CURRENT_TIMESTAMP,
                              DATA_INGESTION_INGESTED_DATA_DIR_KEY,
                              TEST_DATA_DIR_KEY)


# Data Transformation Steps

PREPROCESSING_OBJ_FILE = os.path.join(ROOT_DIR,
                                      ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,
                                      DATA_PREPROCESSED_DIR,
                                      DATA_TRANSFORMATION_PROCESSING_OBJ)

FEATURE_ENGG_OBJ_PATH = os.path.join(ROOT_DIR,
                                     ARTIFACT_DIR_KEY,
                                     DATA_TRANSFORMATION_ARTIFACT,
                                     DATA_PREPROCESSED_DIR,
                                    'feature_engg.pkl')

X_TRANSFORMED_TRAIN_FILE_PATH = os.path.join(ROOT_DIR,
                                           ARTIFACT_DIR_KEY,
                                           DATA_TRANSFORMATION_ARTIFACT,
                                           DATA_TRANSFORM_DIR,
                                           X_TRANSFORM_TRAIN_DIR_KEY)

Y_TRANSFORMED_TRAIN_FILE_PATH = os.path.join(ROOT_DIR,
                                           ARTIFACT_DIR_KEY,
                                           DATA_TRANSFORMATION_ARTIFACT,
                                           DATA_TRANSFORM_DIR,
                                           Y_TRANSFORM_TRAIN_DIR_KEY)

X_TRANSFORMED_TEST_FILE_PATH = os.path.join(ROOT_DIR,
                                          ARTIFACT_DIR_KEY,
                                          DATA_TRANSFORMATION_ARTIFACT,
                                          DATA_TRANSFORM_DIR,
                                          X_TRANSFORM_TEST_DIR_KEY)

Y_TRANSFORMED_TEST_FILE_PATH = os.path.join(ROOT_DIR,
                                          ARTIFACT_DIR_KEY,
                                          DATA_TRANSFORMATION_ARTIFACT,
                                          DATA_TRANSFORM_DIR,
                                          Y_TRANSFORM_TEST_DIR_KEY)

# Model Trainer

MODEL_FILE_PATH = os.path.join(ROOT_DIR,
                               ARTIFACT_DIR_KEY,
                               MODEL_TRAINER_DIR,
                               MODEL_OBJ)


# Batch Prediction

BATCH_PREDICTION_FILE_PATH = os.path.join(ROOT_DIR,
                                          ARTIFACT_DIR_KEY,
                                          BATCH_PREDICTION_DIR,
                                          BATCH_PREDICTION_OUTPUT_DIR_KEY,
                                          BATCH_PREDICTION_OUTPUT_FILE_KEY)

                                          
BATCH_PREDICTION_DATA_PATH = os.path.join(ROOT_DIR,
                                          ARTIFACT_DIR_KEY,
                                          BATCH_PREDICTION_DIR,
                                          BATCH_PREDICTION_DATA_DIR_KEY)

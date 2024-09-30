import os,sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import *
from src.config.configuration import *
from sklearn.pipeline import Pipeline

# PREDICTION_FOLDER = "batch_prediction"
# PREDICTION_CSV = "prediction_csv"
# PREDICTION_FILE = "output.csv"


# ROOT_DIR = os.getcwd()
# BATCH_PREDICTION = os.path.join(ROOT_DIR, PREDICTION_FOLDER,PREDICTION_CSV)



class batch_prediction:
    def __init__(self, input_file_path,
                 model_file_path,
                 processor_file_path,
                 feature_engg_file_path)-> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.processor_file_path = processor_file_path

    def initiate_batch_predictions(self):
        try:

            # Loading data processing pipeline path
            with open(self.processor_file_path,'rb') as f:
                processor = pickle.load(f)

            # Loading model separately
            model = load_model(file_path=self.model_file_path)


            df = pd.read_csv(self.input_file_path)

            df.to_csv("sibros_anomaly_detection.csv")

            
            df.drop(['timestamps','Anomaly_Score'], axis=1)


            # Apply data transformation pipeline steps
            df = processor.transform(df)
            file_path = os.path.join(BATCH_PREDICTION_DATA_PATH,'processedData.csv')
            df.to_csv(file_path, index=False)

            # Make predictionsn
            predictions = model.predict(df)

            df_prediction = pd.DataFrame(predictions,columns=['predictions']).to_csv()

            BATCH_PREDICTION_PATH = BATCH_PREDICTION_FILE_PATH
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'output.csv')
            
            
            df_prediction.to_csv(csv_path, index=False)
            logging.info("Batch prediction done")




        except Exception as e:
            raise CustomException(e, sys)
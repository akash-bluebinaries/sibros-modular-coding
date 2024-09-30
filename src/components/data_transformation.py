import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import *
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import os,sys
from src.config.configuration import *
from dataclasses import dataclass        

@dataclass
class DataTransformationConfig:
    processed_obj_file_path = PREPROCESSING_OBJ_FILE
    X_transformed_train_path = X_TRANSFORMED_TRAIN_FILE_PATH
    y_target_train_path = Y_TRANSFORMED_TRAIN_FILE_PATH
    X_transformed_test_path = X_TRANSFORMED_TEST_FILE_PATH
    y_target_test_path = Y_TRANSFORMED_TEST_FILE_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_variables = [
                'BMS_state',
                'BMS_max_cell_temp_id',
                'BMS_min_cell_temp_id',
                'BMS_max_cell_voltage_id',
                'BMS_min_cell_voltage_id',
                'OBC_mux',
                'OBC_port_status',
                'OBC_overvoltage_fault',
                'OBC_overcurrent_fault',
                'OBC_port_weld_fault'
                ]

            numerical_variables = ['BMS_soc','BMS_soh', 'BMS_bus_voltage', 'BMS_bus_current', 
                                   'BMS_isolation', 'BMS_max_cell_temp', 'BMS_min_cell_temp',
                                   'BMS_max_cell_voltage', 'BMS_min_cell_voltage', 'LV_soc',
                                   'LV_soh', 'LV_voltage', 'LV_current', 'LV_temperature',
                                   'MCU_motor_speed', 'MCU_motor_avg_temp', 'OBC_output_voltage',
                                   'OBC_output_current','OBC_internal_voltage','OBC_internal_current'
                                   ]
            
            numeric_transformer = Pipeline(steps=[
                 ('imputer', SimpleImputer(strategy='mean')),
                 ('scaler', StandardScaler())
                 ])

            # Categorical transformations: Impute missing and OneHotEncode

            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

            # Use ColumnTransformer to apply transformations

            preprocessor = ColumnTransformer(
                transformers=[
                ('num', numeric_transformer, numerical_variables),
                ('cat', categorical_transformer, categorical_variables)]
                )
            save_object(file_path= self.data_transformation_config.processed_obj_file_path,obj=preprocessor)
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading train & test set")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            

            train_df.drop(['timestamps','Anomaly_Score'], axis=1)
            train_df['BMS_max_cell_temp_id'] = train_df['BMS_max_cell_temp_id'].astype('object')
            train_df['BMS_state'] = train_df['BMS_state'].astype('object')
            train_df['BMS_min_cell_temp_id'] = train_df['BMS_min_cell_temp_id'].astype('object')
            train_df['BMS_max_cell_voltage_id'] = train_df['BMS_max_cell_voltage_id'].astype('object')
            train_df['BMS_min_cell_voltage_id'] = train_df['BMS_min_cell_voltage_id'].astype('object')
            train_df['OBC_mux'] = train_df['OBC_mux'].astype('object')
            train_df['OBC_port_status'] = train_df['OBC_port_status'].astype('object')
            train_df['OBC_overvoltage_fault'] = train_df['OBC_overvoltage_fault'].astype('object')
            train_df['OBC_overcurrent_fault'] = train_df['OBC_overcurrent_fault'].astype('object')
            train_df['OBC_port_weld_fault'] = train_df['OBC_port_weld_fault'].astype('object')

            test_df.drop(['timestamps','Anomaly_Score'], axis=1)
            test_df['BMS_max_cell_temp_id'] = test_df['BMS_max_cell_temp_id'].astype('object')
            test_df['BMS_state'] = test_df['BMS_state'].astype('object')
            test_df['BMS_min_cell_temp_id'] = test_df['BMS_min_cell_temp_id'].astype('object')
            test_df['BMS_max_cell_voltage_id'] = test_df['BMS_max_cell_voltage_id'].astype('object')
            test_df['BMS_min_cell_voltage_id'] = test_df['BMS_min_cell_voltage_id'].astype('object')
            test_df['OBC_mux'] = test_df['OBC_mux'].astype('object')
            test_df['OBC_port_status'] = test_df['OBC_port_status'].astype('object')
            test_df['OBC_overvoltage_fault'] = test_df['OBC_overvoltage_fault'].astype('object')
            test_df['OBC_overcurrent_fault'] = test_df['OBC_overcurrent_fault'].astype('object')
            test_df['OBC_port_weld_fault'] = test_df['OBC_port_weld_fault'].astype('object')

            train_df = train_df.drop(['Anomaly_Score','timestamps'], axis = 1)
            test_df = test_df.drop(['Anomaly_Score','timestamps'], axis = 1)


            X_train = train_df.drop("Anomaly", axis=1)
            y_train = train_df['Anomaly']

            X_test = test_df.drop("Anomaly", axis=1)
            y_test = test_df["Anomaly"]
            

            preprocessor = self.get_data_transformer_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)


            # logging.info("Concatenating Independent & Target variable after preprocessing")
            # train_arr = np.c_[X_train, np.array(y_train)]
            # test_arr = np.c_[X_test, np.array(y_test)]


            # logging.info("Converting numpy array to pandas dataframe and saving to Artifact/data_transformation/transformation directory")
            # df_train = pd.DataFrame(train_arr)
            # df_test = pd.DataFrame(test_arr)

            # os.makedirs(os.path.dirname(self.data_transformation_config.X_transformed_train_path), exist_ok=True)
            # X_train_arr.to_csv(self.data_transformation_config.X_transformed_train_path, index=False)

            # os.makedirs(os.path.dirname(self.data_transformation_config.y_target_train_path_transformed_train_path), exist_ok=True)
            # y_train.to_csv(self.data_transformation_config.y_target_train_path, index=False)

            # os.makedirs(os.path.dirname(self.data_transformation_config.X_TRANSFORMED_TEST_FILE_PATH), exist_ok=True)
            # X_test.to_csv(self.data_transformation_config.X_TRANSFORMED_TEST_FILE_PATH, index=False)

            # os.makedirs(os.path.dirname(self.data_transformation_config.Y_TRANSFORMED_TEST_FILE_PATH), exist_ok=True)
            # y_test.to_csv(self.data_transformation_config.Y_TRANSFORMED_TEST_FILE_PATH, index=False)

            print("Transformation Successful")

            return(
                X_train_arr,
                y_train,
                X_test_arr,
                y_test,
                self.data_transformation_config.processed_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        



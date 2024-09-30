from src.config import *
from src.logger import logging
from src.exception import CustomException
import sys
from src.config.configuration import *
from src.utils import load_model
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = PREPROCESSING_OBJ_FILE
            model_path = MODEL_FILE_PATH

            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)

            # Apply preprocessing steps
            data_scaled = preprocessor.transform([features])

            # Make prediction
            pred = model.predict(data_scaled)

            return pred[0]

        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise CustomException(e,sys)
        
"""
categorical_variables = ['BMS_state',
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

numerical_variables = 
['BMS_soc','BMS_soh', 'BMS_bus_voltage', 'BMS_bus_current',
'BMS_isolation', 'BMS_max_cell_temp', 'BMS_min_cell_temp',
'BMS_max_cell_voltage', 'BMS_min_cell_voltage', 'LV_soc',
'LV_soh', 'LV_voltage', 'LV_current', 'LV_temperature',
'MCU_motor_speed', 'MCU_motor_avg_temp', 'OBC_output_voltage',
'OBC_output_current','OBC_internal_voltage','OBC_internal_current']
"""

class CustomData:
    def __init__(self,
                 BMS_state: str,
                 BMS_max_cell_temp_id: str,
                 BMS_max_cell_voltage_id: str,
                 BMS_min_cell_voltage_id: str,
                 OBC_mux: str,
                 OBC_port_status: str,
                 OBC_overvoltage_fault: str,
                 OBC_overcurrent_fault: str,
                 OBC_port_weld_fault: str,


                BMS_soc:float,
                BMS_soh:float,
                BMS_bus_voltage:float,
                BMS_bus_current:float,
                BMS_isolation:float,
                BMS_max_cell_temp:float,
                BMS_min_cell_temp:float,
                BMS_max_cell_voltage:float,
                BMS_min_cell_voltage:float,
                LV_soc:float,
                LV_soh:float,
                LV_voltage:float,
                LV_current:float,
                LV_temperature:float,
                MCU_motor_speed:float,
                MCU_motor_avg_temp:float,
                OBC_output_voltage:float,
                OBC_output_current:float,
                OBC_internal_voltage:float,
                OBC_internal_current:float
                 ):
        
        self.BMS_state = BMS_state
        self.BMS_max_cell_temp_id = BMS_max_cell_temp_id
        self.BMS_max_cell_voltage_id = BMS_max_cell_voltage_id
        self.BMS_min_cell_voltage_id = BMS_min_cell_voltage_id
        self.OBC_mux = OBC_mux
        self.OBC_port_status = OBC_port_status
        self.OBC_overvoltage_fault = OBC_overvoltage_fault
        self.OBC_overcurrent_fault = OBC_overcurrent_fault
        self.OBC_port_weld_fault = OBC_port_weld_fault
        self.BMS_soc = BMS_soc
        self.BMS_soh = BMS_soh
        self.BMS_bus_voltage = BMS_bus_voltage
        self.BMS_bus_current = BMS_bus_current
        self.BMS_isolation = BMS_isolation
        self.BMS_max_cell_temp = BMS_max_cell_temp
        self.BMS_min_cell_temp = BMS_min_cell_temp
        self.BMS_max_cell_voltage = BMS_max_cell_voltage
        self.BMS_min_cell_voltage = BMS_min_cell_voltage
        self.LV_soc = LV_soc
        self.LV_soh = LV_soh
        self.LV_voltage = LV_voltage
        self.LV_current = LV_current
        self.LV_temperature = LV_temperature
        self.MCU_motor_speed = MCU_motor_speed
        self.MCU_motor_avg_temp = MCU_motor_avg_temp
        self.OBC_output_voltage = OBC_output_voltage
        self.OBC_output_current = OBC_output_current
        self.OBC_internal_voltage = OBC_internal_voltage
        self.OBC_internal_current = OBC_internal_current


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict= {
                'BMS_state': self.BMS_state,
                'BMS_max_cell_temp_id': self.BMS_max_cell_temp_id,
                'BMS_max_cell_voltage_id': self.BMS_max_cell_voltage_id,
                'BMS_min_cell_voltage_id': self.BMS_min_cell_voltage_id,
                'OBC_mux': self.OBC_mux,
                'OBC_port_status': self.OBC_port_status,
                'OBC_overvoltage_fault': self.OBC_overvoltage_fault,
                'OBC_overcurrent_fault': self.OBC_overcurrent_fault,
                'OBC_port_weld_fault': self.OBC_port_weld_fault,
                'BMS_soc': self.BMS_soc,
                'BMS_soh': self.BMS_soh,
                'BMS_bus_voltage': self.BMS_bus_voltage,
                'BMS_bus_current': self.BMS_bus_current,
                'BMS_isolation': self.BMS_isolation,
                'BMS_max_cell_temp': self.BMS_max_cell_temp,
                'BMS_min_cell_temp': self.BMS_min_cell_temp,
                'BMS_max_cell_voltage': self.BMS_max_cell_voltage,
                'BMS_min_cell_voltage': self.BMS_min_cell_voltage,
                'LV_soc': self.LV_soc,
                'LV_soh': self.LV_soh,
                'LV_voltage': self.LV_voltage,
                'LV_current': self.LV_current,
                'LV_temperature': self.LV_temperature,
                'MCU_motor_speed': self.MCU_motor_speed,
                'MCU_motor_avg_temp': self.MCU_motor_avg_temp,
                'OBC_output_voltage': self.OBC_output_voltage,
                'OBC_output_current': self.OBC_output_current,
                'OBC_internal_voltage': self.OBC_internal_voltage,
                'OBC_internal_current': self.OBC_internal_current,                
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            logging.info("Error occured in Custom pipeline dataframe")
            raise CustomException(e,sys)

        
                 
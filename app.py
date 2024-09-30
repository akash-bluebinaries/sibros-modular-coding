from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.batch_prediction.batch_pred import *
from werkzeug.utils import secure_filename
from src.pipeline.training_pipeline import Train

feature_engineering_file_path = FEATURE_ENGG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_FILE
model_file_path = MODEL_FILE_PATH


UPLOAD_FOLDER = "batch_prediction/uploaded_csv_file"

app = Flask(__name__, template_folder = "templates")
ALLOWED_EXTENSIONS = {'csv'}

#localhost : 5000
@app.route('/')
def home_page():
    return render_template('index.html')


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

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            BMS_state= int(request.form.get('BMS_state')),
            BMS_max_cell_temp_id= request.form.get('BMS_max_cell_temp_id'),
            BMS_min_cell_temp_id= request.form.get('BMS_min_cell_temp_id'),
            BMS_max_cell_voltage_id= request.form.get('BMS_max_cell_voltage_id'),
            BMS_min_cell_voltage_id= request.form.get('BMS_min_cell_voltage_id'),
            OBC_mux= request.form.get('OBC_mux'),
            OBC_port_status= request.form.get('OBC_port_status'),
            OBC_overvoltage_fault= request.form.get('OBC_overvoltage_fault'),
            OBC_overcurrent_fault= request.form.get('OBC_overcurrent_fault'),
            OBC_port_weld_fault = request.form.get('OBC_port_weld_fault'),
            
            
            BMS_soc= request.form.get('BMS_soc'),
            BMS_soh= request.form.get('BMS_soh'),
            BMS_bus_voltage= request.form.get('BMS_bus_voltage'),
            BMS_bus_current= request.form.get('BMS_bus_current'),
            BMS_isolation= request.form.get('BMS_isolation'),
            BMS_max_cell_temp= request.form.get('BMS_max_cell_temp'),
            BMS_min_cell_temp= request.form.get('BMS_min_cell_temp'),
            BMS_max_cell_voltage= request.form.get('BMS_max_cell_voltage'),
            BMS_min_cell_voltage= request.form.get('BMS_min_cell_voltage'),
            LV_soc= request.form.get('LV_soc'),
            LV_soh= request.form.get('LV_soh'),
            LV_voltage= request.form.get('LV_voltage'),
            LV_current= request.form.get('LV_current'),
            LV_temperature= request.form.get('LV_temperature'),
            MCU_motor_speed= request.form.get('MCU_motor_speed'),
            MCU_motor_avg_temp= request.form.get('MCU_motor_avg_temp'),
            OBC_output_voltage= request.form.get('OBC_output_voltage'),
            OBC_output_current= request.form.get('OBC_output_current'),
            OBC_internal_voltage= request.form.get('OBC_internal_voltage'),
            OBC_internal_current= request.form.get('OBC_internal_current'),
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)


        if pred > 0.5:
            result = "Anomaly"
        else:
            result = "Not Anomaly"

        return render_template('form.html', prediction=result)


@app.route("/batch", methods=["POST","GET"])

def perform_batch_prediction():

    if request.method == "GET":
        return render_template("batch.html")
    
    else:
        file = request.files["csv_file"] # Update the key to 'csv_file'
        # Directory
        directory_path = UPLOAD_FOLDER

        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                        os.remove(file_path)
                
            # Save the new file to the upload directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV Uploaded")
            batch = batch_prediction(file_path,
                                     model_file_path,
                                     transformer_file_path,
                                     feature_engineering_file_path)
            batch.initiate_batch_predictions()

            output = "Batch Prediction Output"

            return render_template("batch.html", prediction_result = output, prediction_type= 'batch')
        else:
            return render_template("batch.html", error = "Invalid File Format", prediction_type= 'batch')
        

@app.route('/train', methods = ['GET','POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()

            return render_template('train.html', success = "Training completed successfully")
        
        except CustomException as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html', error = error_message)
        

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True, port='5000') # 5000

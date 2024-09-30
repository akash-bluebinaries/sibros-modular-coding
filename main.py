from src.logger import logging
from src.components.data_ingestion import *
from src.components.data_transformation import *
from src.components.model_trainer import *




if __name__ == "__main__":
    logging.info("*** Data ingestion started ***")
    obj = DataIngestion()
    train_path, test_path = obj.initate_data_ingestion()
    
    logging.info("*** Data ingestion completed & Data transformation started ***")

    data_transformation = DataTransformation()
    X_train_arr, y_train,X_test_arr, y_test,_ = data_transformation.initiate_data_transformation(train_path, test_path)
    
    logging.info("*** Data transformation completed & Model training started***")

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train_arr, y_train,X_test_arr, y_test)
    
    logging.info("*** Model training completed ***")
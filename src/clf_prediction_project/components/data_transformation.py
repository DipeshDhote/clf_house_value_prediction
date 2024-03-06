import  sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.clf_prediction_project.logger import logging
from src.clf_prediction_project.exception import CustomException
import os

from src.clf_prediction_project.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This fuction responsible for data transformation
        '''
        try:
            numerical_columns = [

                "MedianIncome",
                "MedianAge", 
                "TotalRooms",
                "Total_Bedrooms",
                "Population", 
                "Households", 
                "Latitude", 
                "Longitude",
                "Distancecoast", 
                "DistanceLA", 
                "DistanceSanDiego", 
                "DistanceSanJose",
                "DistanceSanFrancisco"]     
    
            
            # making pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )
             
            logging.info("pipeline for numerical column completed")

            preprocessor = ColumnTransformer(
                
                [   
                  ("NumericalPipeline",num_pipeline,numerical_columns)
                ]
            )
            logging.info("column transformation completed")
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("Reading train and test file")

            preprocessing_obj = self.get_data_transformer_object()
           
            target_column_name="MedianValue" 
            numerical_columns = [

                "MedianIncome",
                "MedianAge", 
                "TotalRooms",
                "Total_Bedrooms",
                "Population", 
                "Households", 
                "Latitude", 
                "Longitude",
                "Distancecoast", 
                "DistanceLA", 
                "DistanceSanDiego", 
                "DistanceSanJose",
                "DistanceSanFrancisco"]                   
        
        
            # divide the train dataset to independent and dependent features

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
                       
            # divide the test dataset to independent and dependent features

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
                
            logging.info("Applying preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            

            logging.info("preprocessing is completed")

            train_arr = np.c_[        
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)




            
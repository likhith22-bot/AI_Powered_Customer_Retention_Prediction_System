import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import warnings
import pickle
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split, GridSearchCV
from missing_values import main_null_values
from varibale_transformation import vt_main
from fs import select_features
from cat_to_num import convert
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from train_models import common
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

class Customer_Retention_System:
    def __init__(self, df):
        try:
            self.df = df.copy()
            logger.info(f"Dataset shape: {self.df.shape}")
            self.df.drop(['customerID'], axis=1, inplace=True)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            logger.info(f"Null values:\n{self.df.isnull().sum()}")
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.y = self.y.map({'Yes': 1, 'No': 0}).astype(int)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f"Training dataset: {self.X_train.shape}")
            logger.info(f"Testing dataset: {self.X_test.shape}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
    def splitting_data_clearly(self):
        try:
            logger.info('\n\n========================================Splitting the data===================================================')
            self.X_train_numerical = self.X_train.select_dtypes(exclude='object')
            self.X_train_categorical = self.X_train.select_dtypes(include="object")
            self.X_test_numerical = self.X_test.select_dtypes(exclude='object')
            self.X_test_categorical = self.X_test.select_dtypes(include="object")

            logger.info(f"Train Numerical columns : {self.X_train_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"Test Numerical columns : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"Train Categorical columns : {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info(f"Test Categorical columns : {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def handling_values(self):
        try:
            logger.info('\n\n======================================Handling NULL values==============================================\n')
            logger.info(f'==================================Training  Data===================================================')
            logger.info(f"Training dataset before handling missing values: \n{self.X_train_numerical.isnull().sum()}")
            self.X_train_numerical=main_null_values(self.X_train_numerical,'Training')
            logger.info(f"Training dataset After handling missing values: \n{self.X_train_numerical.isnull().sum()}")
            logger.info(f'Train data columns name{self.X_train_numerical.columns}')
            logger.info(f'==================================Testing Data===================================================')
            logger.info(f"Testing dataset before handling missing values: \n{self.X_test_numerical.isnull().sum()}")
            self.X_test_numerical=main_null_values(self.X_test_numerical,'Testing')
            logger.info(f"Testing dataset After handling missing values: \n{self.X_test_numerical.isnull().sum()}")
            logger.info(f'Test data columns Names:{self.X_train_numerical.columns}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')

    def var_transformation(self):
        try:
            logger.info('\n\n============================variable transformation===============================================\n')
            logger.info(f'Training data before transformation: {self.X_train_numerical.columns}')
            logger.info(f'Testing data before transformation: {self.X_test_numerical.columns}')
            self.X_train_numerical = vt_main(self.X_train_numerical)
            logger.info(f'Training data After transformation: {self.X_train_numerical.columns}')
            logger.info(f'Testing data After transformation: {self.X_test_numerical.columns}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
    def feature_engineering(self):
        try:
            logger.info('\n\n============================ Feature Selection ===============================================\n')
            logger.info(f'Training data before feature engineering: {self.X_train_numerical.columns}')
            logger.info(f'Testing data before feature engineering: {self.X_test_numerical.columns}')
            self.X_train_numerical,self.X_test_numerical=select_features(self.X_train_numerical,self.X_test_numerical,self.y_train,self.y_test)
            logger.info(f'Training data After feature engineering: {self.X_train_numerical.columns}')
            logger.info(f'Testing data after feature engineering: {self.X_test_numerical.columns}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error line :{er_line.tb_lineno} due to {er_type}:reason : {er_msg}')
    def cat_to_numerical(self):
        try:
            logger.info('\n\n=========================== Categorical to numerical  Selection =============================================\n')
            logger.info(f'Before training data: {self.X_train_categorical.head(10)}')
            logger.info(f'Before training data: {self.X_test_categorical.head(10)}')
            self.X_train_categorical,self.X_test_categorical= convert(self.X_train_categorical,self.X_test_categorical)
            logger.info(f'After training data: {self.X_train_categorical.head(10)}')
            logger.info(f'After training data: {self.X_test_categorical.head(10)}')
            logger.info('\n\n=========================== Final  Selection =============================================\n')
            self.X_train_numerical.reset_index(drop=True, inplace=True)
            self.X_test_numerical.reset_index(drop=True, inplace=True)
            self.X_train_categorical.reset_index(drop=True, inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)
            self.final_train_data=pd.concat([self.X_train_categorical,self.X_train_numerical],axis=1)
            self.final_test_data=pd.concat([self.X_test_categorical,self.X_test_numerical],axis=1)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error line {er_line.tb_lineno} due to {er_type}: reason {er_msg}')

    def data_balancing(self):
        try:
            logger.info(f"Total Final Training data shape : {self.y_train.shape}")
            logger.info(f"Number of Rows for Yes :{1} : Class : {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for No :{0} : Class : {sum(self.y_train == 0)}")
            sm_obj = SMOTE(random_state=42)
            self.X_train_bal, self.y_train_bal = sm_obj.fit_resample(self.final_train_data, self.y_train)
            logger.info(f"After Balancing Training data shape : {self.y_train_bal.shape}")
            logger.info(f"Number of Rows for Yes :{1} : Class : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for No :{0} : Class : {sum(self.y_train_bal == 0)}")
            global sc
            sc = StandardScaler()
            sc.fit(self.X_train_bal)
            self.X_train_bal_scaled = pd.DataFrame(sc.transform(self.X_train_bal),columns=self.final_train_data.columns)
            self.X_test_scaled = pd.DataFrame(sc.transform(self.final_test_data),columns=self.final_test_data.columns)
            logger.info(f"===========Final Train Data============================= : \n : {self.X_train_bal_scaled.shape} : \n : {self.X_train_bal_scaled.columns} : \n : {self.X_train_bal_scaled.head(10)} : \n : {self.X_train_bal_scaled.isnull().sum()}")
            logger.info(f"===========Final Test Data============================= : \n : {self.X_test_scaled.shape} : \n : {self.X_test_scaled.columns} : \n : {self.X_test_scaled.head(10)} : \n : {self.X_test_scaled.isnull().sum()}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def train_all_models(self):
        try:
            #common(self.X_train_bal_scaled,self.y_train_bal,self.X_test_scaled,self.y_test)
            global nb_reg
            nb_reg = GaussianNB()
            nb_reg.fit(self.X_train_bal_scaled, self.y_train_bal)
            logger.info(f"Test Data Accuracy : {accuracy_score(self.y_test, nb_reg.predict(self.X_test_scaled))}")
            logger.info(f"Test Data Confusion Matrix : {confusion_matrix(self.y_test, nb_reg.predict(self.X_test_scaled))}")
            logger.info(f"Test Data Classification Report  : {classification_report(self.y_test, nb_reg.predict(self.X_test_scaled))}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in line no :{er_line.tb_lineno} : due to :{er_type} : reason : {er_msg}')
    def select_best_parameter(self):
        try:
            global grid_search
            param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
            grid_search = GridSearchCV(estimator=nb_reg,param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
            grid_search.fit(self.X_train_bal_scaled, self.y_train_bal)
            logger.info(f"Best Parameters:{ grid_search.best_params_}")
            logger.info(f"Best CV Score:{grid_search.best_score_}")
            logger.info(f"Test Accuracy:{grid_search.best_estimator_.score(self.X_test_scaled, self.y_test)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in line no: {er_line.tb_lineno} due to {er_type} : reason : {er_msg}')
    def save_to_file(self):
        try:
            with open('scaledown.pkl','wb') as f:
                pickle.dump(sc, f)
            with open('Model.pkl','wb') as m:
                pickle.dump(grid_search, m)
            logger.info(f"Model Saved Successfully")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            logger.info(f'Error line in :{er_line.tb_lineno} due to {er_type} : reason :{er_msg}')
    def sample_pridict(self):
        try:
            #logger.info(f'{np.array(self.final_train_data.iloc[1,:])}')
            val = np.array([[0.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,1.00000e+00,1.00000e+00,
                             0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,0.00000e+00,
                             0.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,1.00000e+00,1.00000e+00,
                             0.00000e+00,0.00000e+00,0.00000e+00,2.00000e+00,0.00000e+00,5.40000e+01,9.72000e+01,5.12945e+03]])
            with open('scaledown.pkl','rb') as f:
                scaled = pickle.load(f)
            with open('Model.pkl','rb') as m:
                nb=pickle.load(m)
            logger.info(f"Sample Prediction is {nb.predict(scaled.transform(val))} ")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            logger.info(f'Error line in: {er_line.tb_lineno} due to {er_type} : reason : {er_msg}')
if __name__=='__main__':
    try:
        dfs=pd.read_csv('Data.csv')
        obj=Customer_Retention_System(dfs)
        obj.splitting_data_clearly()
        obj.handling_values()
        obj.var_transformation()
        obj.feature_engineering()
        obj.cat_to_numerical()
        obj.data_balancing()
        obj.train_all_models()
        obj.select_best_parameter()
        obj.save_to_file()
        obj.sample_pridict()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
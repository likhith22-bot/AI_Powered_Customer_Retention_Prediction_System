import numpy as np
import pandas as pd
import sys
from log_code import setup_logging
logger = setup_logging('cat_to_num')
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
def onehot_encode(x_train,x_test):
    try:
        logger.info(f'==========================================one hot encoding===================================')
        logger.info(f'Before one hot training data is:\n{x_train.head(10)}')
        logger.info(f'Before one hot testing data is:\n{x_test.head(10)}')
        one_obj=OneHotEncoder(drop='first')
        one_obj.fit(x_train)
        result = one_obj.transform(x_train).toarray()
        result2 = one_obj.transform(x_test).toarray()
        t = pd.DataFrame(data=result, columns=one_obj.get_feature_names_out())
        t2 = pd.DataFrame(data=result2, columns=one_obj.get_feature_names_out())
        logger.info(f'After one hot training data is:\n{t.head(10)}')
        logger.info(f'After one hot testing data is:\n{t2.head(10)}')
        return t,t2
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
def ordinal_encode(x_train,x_test):
    try:
        logger.info(f'==========================================Ordinal encoding===================================')
        logger.info(f'Before ordinal training data is:\n{x_train.head(10)}')
        logger.info(f'Before ordinal testing data is:\n{x_test.head(10)}')
        ordinal_obj=OrdinalEncoder()
        ordinal_obj.fit(x_train)
        result = ordinal_obj.transform(x_train)
        result2 = ordinal_obj.transform(x_test)
        p1 = pd.DataFrame(data=result, columns=ordinal_obj.get_feature_names_out() + "_res")
        p2 = pd.DataFrame(data=result2, columns=ordinal_obj.get_feature_names_out() + "_res")
        logger.info(f'After ordinal training data is:\n{p1.head(10)}')
        logger.info(f'After ordinal testing data is:\n{p2.head(10)}')
        return p1,p2
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
def frequency_encode(x_train,x_test):
    try:
        logger.info(f'==========================================Frequency encoding===================================')
        logger.info(f'Before frequency training data is:\n{x_train.head(10)}')
        logger.info(f'Before frequency testing data is:\n{x_test.head(10)}')
        for var in x_train.columns:
            fre = x_train[var].value_counts()
            x_train[var] = x_train[var].map(fre)
        for var in x_test.columns:
            fre = x_test[var].value_counts()
            x_test[var] = x_test[var].map(fre)
        logger.info(f'After frequency training data is:\n{x_train.head(10)}')
        logger.info(f'After frequency testing data is:\n{x_test.head(10)}')

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
def thermometer_encode(x_train,x_test):
    try:
        logger.info(f'==========================================Thermometer encoding===================================')
        logger.info(f'Before thermometer training data is:\n{x_train.head(10)}')
        logger.info(f'Before thermometer testing data is:\n{x_test.head(10)}')
        mapping = {
            'Month-to-month': 0,
            'One year': 1,
            'Two year': 2
        }
        x_train['Rank'] = x_train['Contract'].map(mapping)
        for i in range(1, 3):
            x_train[f'Contract_Level_{i}'] = (x_train['Rank'] >= i).astype(int)
        x_train.drop('Rank', axis=1, inplace=True)
        x_test['Rank'] = x_test['Contract'].map(mapping)
        for i in range(1, 3):
            x_test[f'Contract_Level_{i}'] = (x_test['Rank'] >= i).astype(int)
        x_test.drop('Rank', axis=1, inplace=True)
        logger.info(f'After thermometer training data is:\n{x_train.head(10)}')
        logger.info(f'After thermometer testing data is:\n{x_test.head(10)}')
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')
def convert(x_train,x_test):
    try:
        ord1=pd.DataFrame(x_train['Contract'])
        ord2=pd.DataFrame(x_test['Contract'])
        x_train.drop('Contract', axis=1, inplace=True)
        x_test.drop('Contract', axis=1, inplace=True)
        temp1,temp2=onehot_encode(x_train,x_test)
        t1,t2=ordinal_encode(ord1,ord2)
        final1=pd.concat([temp1,t1],axis=1)
        final2=pd.concat([temp2,t2],axis=1)
        return final1,final2
        #frequency_encode(x_train,x_test)
        #thermometer_encode(ord1,ord2)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in the line no:{er_line.tb_lineno} due to : {er_type}:reason : {er_msg}')

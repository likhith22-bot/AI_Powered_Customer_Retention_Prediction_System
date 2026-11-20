import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('varibale_transformation')
from scipy.stats import skew, boxcox, yeojohnson


def log_transform(x):
    try:
        return np.log(x)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error line: {er_line.tb_lineno} : due to {er_type}: reason: {er_msg}")


def reciprocal_transform(x):
    try:
        return 1 / (x + 1e-6)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error line: {er_line.tb_lineno} : due to {er_type}: reason: {er_msg}")

def sqrt_transform(x):
    try:
        return np.sqrt(x)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')


def exp_transform(x):
    try:
        return np.exp(x)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')


def boxcox_transform(x):
    try:
        min_val = x.min()
        if min_val <= 0:
            x_shifted = x - min_val + 1e-6
            d, l = boxcox(x_shifted.dropna())
        else:
            d, l = boxcox(x.dropna())
        result = pd.Series(np.nan, index=x.index)
        result.loc[x.dropna().index] = d
        return result
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')


def yeojohnson_transform(x):
    try:
        d, l = yeojohnson(x)

        result = pd.Series(np.nan, index=x.index)
        result.loc[x.index] = d
        return result
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')
def EDA(data,var):
    try:
        plt.figure(figsize=(8, 3))
        plt.title(var)
        plt.subplot(1, 2, 1)
        plt.title("Normal Distribution")
        data[var].plot(kind='kde', color='r')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[var])
        plt.show()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line:{er_line.tb_lineno}:due to {er_type}: reson :{er_msg}')

def checking(data):
    try:
        for var in data.columns:
            EDA(data,var)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')


def vt_main(X):
    try:
        df = pd.DataFrame()
        # for col in X.columns:
        #     df[col + '_log'] = log_transform(X[col])
        #     df[col + '_reci'] = reciprocal_transform(X[col])
        #     df[col + '_sqrt'] = sqrt_transform(X[col])
        #     df[col + '_exp'] = exp_transform(X[col])
        #     df[col + '_boxcox'] = boxcox_transform(X[col])
        #     df[col + '_yeo'] = yeojohnson_transform(X[col])
        #     checking(df)
        #     df.drop(columns=df.columns, inplace=True)
        return X
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line: {er_line.tb_lineno} : due to {er_type}: {er_msg}')
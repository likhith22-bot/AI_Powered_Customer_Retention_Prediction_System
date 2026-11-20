import sys
import numpy as np
import pandas as pd
from log_code import setup_logging
logger = setup_logging('missing_values')
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer

def mean(x):
    try:
        return x.fillna(x.mean())
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def median(x):
    try:
        return x.fillna(x.median())
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def mode(x):
    try:
        return x.fillna(x.mode()[0])
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def arbitary(x):
    try:
        return x.fillna(999)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def constant(x):
    try:
         d=np.where(x.isnull() , 0 , 1)
         return d
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def EOD(x):
    try:
        return x.fillna(x.mean() + 3 * x.std())
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in line {er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def ran_sample(x):
    try:
        t = x.copy()
        null_index = x[x.isnull()].index
        if len(null_index) == 0:
            return t
        s = x.dropna().sample(len(null_index),random_state=42)
        s.index = null_index
        t.loc[null_index] = s
        return t
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line:{er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def KNN_Imputation(x):
    try:
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        if isinstance(x, pd.Series):
            x = x.to_frame()
        result = imputer.fit_transform(x)
        return pd.DataFrame(result,columns=x.columns,index=x.index)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line:{er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def forward_filling(x):
    try:
        return x.fillna(x.ffill())
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line :{er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def check(df,val):
    try:
        logger.info(f'The original data std values is:{val}')
        for i in df.columns:
            logger.info(f'values {i} {df[i].std()}: diffrence{val-df[i].std()}')
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line:{er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')
def main_null_values(dfs,str):
    try:
        df=pd.DataFrame()
        logger.info(f'=================================={str} Data===================================================')
        logger.info(f'NUll values in data before applying the method \n{dfs.isnull().sum()}')
        for X in dfs.columns:
            if dfs[X].isnull().sum() >0:
                # val=dfs[X].std()
                # df[X+'_mean'] = mean(dfs[X])
                # df[X+'_med'] = median(dfs[X])
                df[X+'_mod'] = mode(dfs[X])
                # df[X+'_arb'] = arbitary(dfs[X])
                # df[X+'_cons'] = constant(dfs[X])
                # df[X+'_EOD'] = EOD(dfs[X])
                # df[X+'_KNN_Imputation'] = KNN_Imputation(dfs[X])
                # df[X+'_forward_fill'] = forward_filling(dfs[X])
                # df[X + '_ran_sample'] = ran_sample(dfs[X])
                dfs[X+'_mod']=df[X+'_mod']
                dfs.drop(columns=[X],inplace=True)
                #check(df,val)
        logger.info(f'NUll in the data After applying the method \n {dfs.isnull().sum()}')
        return dfs
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error line:{er_line.tb_lineno}:due to {er_type}:reason:{er_msg}')

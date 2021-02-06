import pandas as pd
from sklearn.datasets import load_boston

def boston_housing_prices():
    """
    Load the Boston Housing Prices dataset and return it as a Pandas Dataframe
    """
    boston = load_boston()
    df = pd.DataFrame(boston['data'] )
    df.columns = boston['feature_names']
    df['PRICE']= boston['target']
    return df

def boston_housing_prices_descr():
    print(load_boston().DESCR)



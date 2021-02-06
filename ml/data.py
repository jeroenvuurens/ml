import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import load_iris, load_boston
from sklearn.utils import resample
from .helper import *
import matplotlib.pyplot as plt

class Data:
    def __init__(self, train_X, train_y, valid_X, valid_y, test_X, test_y, label_X, label_y, degree=1, bias=False, column_y = False, scale=False, f_y=None, balance=False):
        self._train_X = train_X
        self._train_y = train_y
        self._valid_y = valid_y if len(valid_X) > 0 else train_y
        self._valid_X = valid_X if len(valid_X) > 0 else train_X
        self._test_X = test_X
        self._test_y = test_y
        self._label_X = label_X
        self.label_y = label_y
        self._degree = degree
        self._bias = bias
        self._scale = scale
        self.f_y = f_y
        self._column_y = column_y
        self._balance = balance

    def train(self):
        return zip(self.train_X, self.train_y)

    def valid(self):
        return zip(self.valid_X, self.valid_y)

    def test(self):
        return zip(self.test_X, self.test_y)

    def thresholds(self, t):
        train_y = thresholds(self.train_y, t)
        valid_y = thresholds(self.valid_y, t)
        test_y = thresholds(self.test_y, t)
        return Data(self._train_X, train_y, self._valid_X, valid_y, self._test_X, test_y, self.label_X, self.label_y, degree=self.degree, bias=self.bias)

    @property
    def column_y(self):
        return self._column_y

    @column_y.setter
    def column_y(self, value):
        try:
            del self._train_ty
        except: pass
        try:
            del self._valid_ty
        except: pass
        try:
            del self._test_ty
        except: pass
        self._column_y = value

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = value
        try:
            del self._train_tX
        except: pass
        try:
            del self._valid_tX
        except: pass
        try:
            del self._test_tX
        except: pass

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value
        try:
            del self._train_tX
        except: pass
        try:
            del self._valid_tX
        except: pass
        try:
            del self._test_tX
        except: pass
        try:
            del self._label_tX
        except: pass

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        try:
            del self._train_tX
        except: pass
        try:
            del self._valid_tX
        except: pass
        try:
            del self._test_tX
        except: pass

    def polyX(self, X):
        if self.degree > 1:
            return polynomials(X, self.degree)
        else:
            return X

    def scaleX(self, X):
        X = self.polyX(X)
        if self._scale:
            try:
                self._scaler
            except:
                self._scaler = StandardScaler()
                self._scaler.fit(self.polyX(self._train_X))
            return self._scaler.transform(X)
        else:
            return X

    def transformed_y(self, y):
        y = np.asarray(y)
        if self.f_y is not None:
            y = self.f_y(y)
        return y

    def transform_y(self, y):
        y = self.transformed_y(y)
        if self.column_y:
            y = np.expand_dims(y, axis=1)
        return y

    def indices(self, y):
        if self._balance:
            indices = [np.where(y==l)[0] for l in np.unique(y)]
            classlengths = [len(i) for i in indices]
            n = max(classlengths)
            mask = np.hstack([np.random.choice(i, n-l, replace=True) for l,i in zip(classlengths, indices)])
            return np.hstack([mask, range(len(y))])
        return list(range(len(y)))

    @property
    def train_indices(self):
        try:
            return self._train_indices
        except:
            self._train_indices = self.indices(self._train_y)
            try:
                if self._resample == 0:
                    self._resample = len(self._train_indices)
                if self._resample > 0:
                    self._train_indices = resample(self._train_indices, n_samples = self._resample)
            except: pass
            return self._train_indices

    def resample(self, n=0):
        self._resample = n
        try:
            del self._train_indices
            del self._train_tX
            del self._train_ty
        except: pass

    def noresample(self):
        self.resample(None)

    @property
    def label_X(self):
        try:
            return self._label_tX
        except:
            if self.bias:
                self._label_tX = ['bias']
                self._label_tX.extend(self._label_X)
            else:
                self._label_tX = self._label_X
            return self._label_tX

    def transform_X(self, X):
        X = self.scaleX(X)
        if self.bias:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    @property
    def train_X(self):
        try:
            return self._train_tX
        except:
            self._train_tX = self.transform_X(self._train_X)[self.train_indices]
            return self._train_tX

    @property
    def valid_X(self):
        try:
            return self._valid_tX
        except:
            self._valid_tX = self.transform_X(self._valid_X)
            return self._valid_tX

    @property
    def test_X(self):
        try:
            return self._test_tX
        except:
            self._test_tX = self.transform_X(self._test_X)
            return self._test_tX

    def train_X_interpolated(self, factor=2):
        if factor == 0:
            return self.train_X
        X = order_x(self._train_X)
        for i in range(factor):
            X = np.append( X, np.array([ (x1 + x2) / 2 for x1, x2 in zip(X[:-1], X[1:]) ] ), axis=0)
            X = order_x(X)
        return self.transform_X(X)

    def valid_X_interpolated(self, factor=2):
        if factor == 0:
            return self.valid_X
        X = order_x(self._valid_X)
        for i in range(factor):
            X = np.append( X, np.array([ (x1 + x2) / 2 for x1, x2 in zip(X[:-1], X[1:]) ] ), axis=0)
            X = order_x(X)
        return self.transform_X(X)

    def test_X_interpolated(self, factor=2):
        if factor == 0:
            return self.test_X
        X = order_x(self._test_X)
        for i in range(factor):
            X = np.append( X, np.array([ (x1 + x2) / 2 for x1, x2 in zip(X[:-1], X[1:]) ] ), axis=0)
            X = order_x(X)
        return self.transform_X(X)

    def X_interpolated(self, factor=2):
        X = order_x(np.vstack([self._train_X, self._valid_X]))
        if factor == 0:
            return self.X
        for i in range(factor):
            X = np.append( X, np.array([ (x1 + x2) / 2 for x1, x2 in zip(X[:-1], X[1:]) ] ), axis=0)
            X = order_x(X)
        return self.transform_X(X)


    @property
    def train_y(self):
        try:
            return self._train_ty
        except:
            self._train_ty = self.transform_y(self._train_y)[self.train_indices]
            return self._train_ty

    @property
    def valid_y(self):
        try:
            return self._valid_ty
        except:
            self._valid_ty = self.transform_y(self._valid_y)
            return self._valid_ty

    @property
    def test_y(self):
        try:
            return self._test_ty
        except:
            self._test_ty = self.transform_y(self._test_y)
            return self._test_ty

    def _plot(self, x, y = None, xlabel='', ylabel=None, marker='.' ):
        if y is None:
            y = self.train_y
        plt.scatter(x, y, marker=marker)
        if ylabel is None:
            plt.ylabel(self.label_y)
        else:
            plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    def plot(self, x = 0, y = None, xlabel = None, **kwargs ):
        if xlabel is None:
            xlabel = self.label_X[x]
        self._plot( self.train_X[:, x], y, xlabel, **kwargs )

    def plot_valid(self, x = 0, y = None, xlabel = None, **kwargs ):
        if xlabel is None:
            xlabel = self.label_X[x]
        self._plot( self.valid_X[:, x], self.valid_y, xlabel, **kwargs )

    def _plot2d(self, X, y, markersize=4, x1=0, x2=1, loc='upper right', noise_x1=0, noise_x2=0):
        for c in sorted(np.unique(y)):
            x = X[(c == y).flatten()]
            xx1 = x[:, x1]
            xx2 = x[:, x2] 
            if noise_x1 > 0:
                xx1 = xx1 + np.random.normal(0, noise_x1, xx1.shape)
            if noise_x2 > 0:
                xx2 = xx2 + np.random.normal(0, noise_x2, xx2.shape)
            plt.plot(xx1, xx2, '.', markersize=markersize, label=int(c))
        plt.ylabel(self.label_X[x2])
        plt.xlabel(self.label_X[x1])
        plt.gca().legend(loc=loc)

    def plot2d_valid(self, markersize=4, x1=0, x2=1, loc='upper right', noise_x1=0, noise_x2=0):
        self._plot2d(self.valid_X, self.valid_y, markersize=markersize, x1=x1, x2=x2, loc=loc, noise_x1=noise_x1, noise_x2=noise_x2)

    def plot2d(self, markersize=4, x1=0, x2=1, loc='upper right', noise_x1=0, noise_x2=0):
        self._plot2d(self.train_X, self.train_y, markersize=markersize, x1=x1, x2=x2, loc=loc, noise_x1=noise_x1, noise_x2=noise_x2)

    @classmethod
    def from_dataframe(cls, dataframe, target, *features, **kwargs):
        features = [f for f in features if f is not None]
        if len(features) == 0:
            features = dataframe.columns.difference([target])
        X = np.array(dataframe[features]) 
        y = dataframe[target]
        return cls.from_numpy(X, y, target, features, **kwargs)

    @classmethod
    def from_numpy(cls, X, y, target, features, valid_perc=0.2, test_perc=0.0, random_state=3, **kwargs):
        if test_perc > 0:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_perc, random_state=random_state)
        else:
            X_test = None
            y_test = None
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_perc / (1 - test_perc), random_state=random_state)
        return cls(X_train, y_train, X_valid, y_valid, X_test, y_test, features, target, **kwargs)

def thresholds(y, thresholds):
    yn = np.zeros(y.shape)
    for t in thresholds:
        yn += (y >= t) * 1
    return yn

def autompg_pd():
    df = pd.read_csv('/data/datasets/auto-mpg.csv', delimiter=',')
    df = df[df['horsepower'] != '?']
    df['horsepower'] = df['horsepower'].astype(int)
    return df

def autompg(*features, **kwargs):
    df = autompg_pd()
    if len(features) == 0:
        features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
    return Data.from_dataframe( df, 'mpg', *features, **kwargs )

def bigmart_pd():
    df = pd.read_csv('/data/datasets/bigmartdatasales.csv', delimiter=',')
    return df

def bigmart(*features, **kwargs):
    df = bigmart_pd()
    return Data.from_dataframe( df, 'Item_Outlet_Sales', *features, **kwargs )

def advertising_pd():
    df = pd.read_csv('/data/datasets/advertising.csv', delimiter=',')
    # this is fiction, just pretend that we can compute something as profit
    # based on sales - costs for the purpose of demonstrating polynomial regression
    df['Profit'] = df.Sales * 20 - df.TV - df.Radio - df.Newspaper - 50
    return df

def advertising(target, *features, **kwargs):
    transform_y = kwargs.pop('transform_y', None)
    df = advertising_pd()
    df = df.drop(columns=['Sales' if target == 'Profit' else 'Profit'])
    data = Data.from_dataframe( df, target, *features, **kwargs )
    if transform_y is not None:
        return data.transform_y(transform_y)
    return data

def advertising_sales_tv(**kwargs):
    return advertising('Sales', 'TV', **kwargs)

def advertising_profit_tv(**kwargs):
    return advertising('Profit', 'TV', **kwargs)

def advertising_profit_classify(**kwargs):
    return advertising('Profit', 'TV', 'Radio', transform_y = lambda y: (y > 0) * 1)

def advertising_sales_radio(**kwargs):
    return advertising('Sales', 'Radio', **kwargs)

def advertising_sales_newspaper(**kwargs):
    return advertising('Sales', 'Newspaper', **kwargs)

def advertising_sales(**kwargs):
    return advertising('Sales', **kwargs)

def iris_pd():
    iris=load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
    return df

def boston_pd():
    boston=load_boston()
    features = np.array([x.lower() for x in boston.feature_names])
    df = pd.DataFrame(data=boston.data, columns=features)
    df['price'] = pd.Series(boston.target)
    return df

def boston_lstat(**kwargs):
    return Data.from_dataframe(boston_pd(), 'price', 'lstat', **kwargs)

def boston(*features, **kwargs):
    return Data.from_dataframe(boston_pd(), 'price', *features, **kwargs)

def iris_binary_pd():
    df = iris_pd()
    df = df[df.target > 0]
    df.target -= 1
    return df

def iris_classify(**kwargs):
    return Data.from_dataframe( iris_binary_pd(), 'target', 'petal length (cm)', 'petal width (cm)', **kwargs)

def iris(**kwargs):
        return Data.from_dataframe( iris_binary_pd(), 'target', **kwargs)

def titanic_pd():
    df = pd.read_csv('/data/datasets/titanic.csv', delimiter=',')
    df.Sex = (df.Sex == 'male') * 1
    df = df[['Survived', 'Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare']]
    df = df.dropna()
    return df

def titanic(*features, **kwargs):
    return Data.from_dataframe( titanic_pd(), 'Survived', *features, **kwargs )

def liver_pd():
    df = pd.read_csv('/data/datasets/indian_liver_patient.csv', delimiter=',')
    df.Gender = (df.Gender == 'Male') * 1
    df = df.rename(columns={'Dataset':'Disease'})
    df.Disease = df.Disease - 1
    df = df.dropna()
    return df

def liver(*features, **kwargs):
    return Data.from_dataframe( liver_pd(), 'Disease', *features, **kwargs )

def wines_pd():
    return pd.read_csv('/data/datasets/winequality-red.csv', delimiter=';')

def wines_binary(target, *features, threshold=6, **kwargs):
    return wines(target, *features, f_y=lambda y: (y >= threshold) * 1, **kwargs)

def wines_multi_class(target, thresholds, *features, **kwargs):
    return wines(target, *features, **kwargs).thresholds(thresholds)

def wines(target, *features, **kwargs):
    transform_y = kwargs.pop('transform_y', None)
    data = Data.from_dataframe( wines_pd(), target, *features, **kwargs )
    if transform_y is not None:
        return data.transform_y(transform_y)
    return data

def wines_quality_alcohol(**kwargs):
    return wines('quality', 'alcohol', **kwargs)

def polynomials(X, degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)

def dam(**kwargs):
    with open("/data/datasets/dam_water_data.pickle", "rb") as myfile:
        X_train, X_val, X_test, X_all, y_train, y_val, y_test, y_all = pickle.load(myfile)
    target = 'Hydrostatics of a dam'
    features = ['Outflow of water']

    if 'random_state' in kwargs or 'valid_perc' in kwargs:
        return Data.from_numpy(X_all, y_all, target, features, **kwargs)
    return Data(X_train, y_train, X_val, y_val, X_test, y_test, ['Outflow of water'], 'Hydrostatics of a dam', **kwargs)

def speeddating_pd():
    df = pd.read_csv('/data/datasets/Speed Dating.csv', encoding="ISO-8859-1" )
    df = df[(df.wave < 6) | (df.wave > 9)]
    df = df.drop(columns=['id', 'wave', 'round', 'position', 'positin1', 'int_corr', 'idg', 'partner','samerace', 'prob', 'prob_o', 'met', 'met_o', 'order', 'match_es', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'met', 'length', 'you_call', 'them_cal'])
    df = df.drop(columns=[c for c in df.columns if '_o' in c])
    df = df.drop(columns=[c for c in df.columns if '_2' in c])
    df = df.drop(columns=[c for c in df.columns if '_3' in c])
    df = df[df.pid.notna()]
    df = df.merge(df, left_on=[df.iid, df.pid], right_on=[df.pid, df.iid], suffixes=['_x', '_y'])
    df = df[df.gender_y == 0]
    df['match'] = df.match_x
    df = df.drop(columns=['key_0', 'key_1', 'match_x', 'match_y'])
    return df

def speeddating_old_pd():
    df = pd.read_csv('/data/datasets/Speed Dating.csv', encoding="ISO-8859-1" )
    df = df[(df.wave < 6) | (df.wave > 11)]
    df = df.drop(columns=['id', 'wave', 'round', 'position', 'positin1', 'int_corr', 'dec', 'dec_o', 'idg', 'partner','samerace', 'prob', 'prob_o', 'met', 'met_o', 'order', 'match_es'])
    df = df.drop(columns=[c for c in df.columns if '_o' in c])
    df = df.drop(columns=[c for c in df.columns if '2' in c])
    df = df.drop(columns=[c for c in df.columns if '4' in c])
    df = df[df.pid.notna()]
    df = df.merge(df, left_on=[df.iid, df.pid], right_on=[df.pid, df.iid], suffixes=['_x', '_y'])
    df = df[df.gender_y == 0]
    df['match'] = df.match_x
    df = df.drop(columns=['key_0', 'key_1', 'match_x', 'match_y'])
    return df

def heart_disease_pd():
    columns = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest',
          'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs',
          'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo',
          'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic',
          'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest',
          'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang',
          'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca',
          'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm',
          'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday',
          'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain',
          'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2',
          'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']
    with open('/data/datasets/cleveland.data') as fin:
        current = []
        table = []
        for line in fin:
            current.extend(line.split())
            if 'name' in line:
                table.append(current)
                current = []
        df = pd.DataFrame(table, columns=columns)
    for c in df.columns:
        if c == 'thaldur' or c == 'thaltime' or c == 'met' or c == 'oldpeak':
            df[c] = df[c].apply(float)
        elif c == 'name': pass
        elif c == 'sex':
            df[c] = df[c].replace({'0': 'F', '1':'M'})
        else:
            df[c] = df[c].apply(int)
    return df

def communityviolence_desc():
    return """
-- communityname: Community name - not predictive - for information only (string) 
-- state: US state (by 2 letter postal abbreviation)(nominal) 
-- countyCode: numeric code for county - not predictive, and many missing values (numeric) 
-- communityCode: numeric code for community - not predictive and many missing values (numeric) 
-- fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests - not predictive (numeric - integer) 

-- population: population for community: (numeric - expected to be integer) 
-- householdsize: mean people per household (numeric - decimal) 
-- racepctblack: percentage of population that is african american (numeric - decimal) 
-- racePctWhite: percentage of population that is caucasian (numeric - decimal) 
-- racePctAsian: percentage of population that is of asian heritage (numeric - decimal) 
-- racePctHisp: percentage of population that is of hispanic heritage (numeric - decimal) 
-- agePct12t21: percentage of population that is 12-21 in age (numeric - decimal) 
-- agePct12t29: percentage of population that is 12-29 in age (numeric - decimal) 
-- agePct16t24: percentage of population that is 16-24 in age (numeric - decimal) 
-- agePct65up: percentage of population that is 65 and over in age (numeric - decimal) 
-- numbUrban: number of people living in areas classified as urban (numeric - expected to be integer) 
-- pctUrban: percentage of people living in areas classified as urban (numeric - decimal) 
-- medIncome: median household income (numeric - may be integer) 
-- pctWWage: percentage of households with wage or salary income in 1989 (numeric - decimal) 
-- pctWFarmSelf: percentage of households with farm or self employment income in 1989 (numeric - decimal) 
-- pctWInvInc: percentage of households with investment / rent income in 1989 (numeric - decimal) 
-- pctWSocSec: percentage of households with social security income in 1989 (numeric - decimal) 
-- pctWPubAsst: percentage of households with public assistance income in 1989 (numeric - decimal) 
-- pctWRetire: percentage of households with retirement income in 1989 (numeric - decimal) 
-- medFamInc: median family income (differs from household income for non-family households) (numeric - may be integer) 
-- perCapInc: per capita income (numeric - decimal) 
-- whitePerCap: per capita income for caucasians (numeric - decimal) 
-- blackPerCap: per capita income for african americans (numeric - decimal) 
-- indianPerCap: per capita income for native americans (numeric - decimal) 
-- AsianPerCap: per capita income for people with asian heritage (numeric - decimal) 
-- OtherPerCap: per capita income for people with 'other' heritage (numeric - decimal) 
-- HispPerCap: per capita income for people with hispanic heritage (numeric - decimal) 
-- NumUnderPov: number of people under the poverty level (numeric - expected to be integer) 
-- PctPopUnderPov: percentage of people under the poverty level (numeric - decimal) 
-- PctLess9thGrade: percentage of people 25 and over with less than a 9th grade education (numeric - decimal) 
-- PctNotHSGrad: percentage of people 25 and over that are not high school graduates (numeric - decimal) 
-- PctBSorMore: percentage of people 25 and over with a bachelors degree or higher education (numeric - decimal) 
-- PctUnemployed: percentage of people 16 and over, in the labor force, and unemployed (numeric - decimal) 
-- PctEmploy: percentage of people 16 and over who are employed (numeric - decimal) 
-- PctEmplManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal) 
-- PctEmplProfServ: percentage of people 16 and over who are employed in professional services (numeric - decimal) 
-- PctOccupManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal) #### No longer sure of difference from PctEmplManu - may include unemployed manufacturing workers #### 
-- PctOccupMgmtProf: percentage of people 16 and over who are employed in management or professional occupations (numeric - decimal) 
-- MalePctDivorce: percentage of males who are divorced (numeric - decimal) 
-- MalePctNevMarr: percentage of males who have never married (numeric - decimal) 
-- FemalePctDiv: percentage of females who are divorced (numeric - decimal) 
-- TotalPctDiv: percentage of population who are divorced (numeric - decimal) 
-- PersPerFam: mean number of people per family (numeric - decimal) 
-- PctFam2Par: percentage of families (with kids) that are headed by two parents (numeric - decimal) 
-- PctKids2Par: percentage of kids in family housing with two parents (numeric - decimal) 
-- PctYoungKids2Par: percent of kids 4 and under in two parent households (numeric - decimal) 
-- PctTeen2Par: percent of kids age 12-17 in two parent households (numeric - decimal) 
-- PctWorkMomYoungKids: percentage of moms of kids 6 and under in labor force (numeric - decimal) 
-- PctWorkMom: percentage of moms of kids under 18 in labor force (numeric - decimal) 
-- NumKidsBornNeverMar: number of kids born to never married (numeric - expected to be integer) 
-- PctKidsBornNeverMar: percentage of kids born to never married (numeric - decimal) 
-- NumImmig: total number of people known to be foreign born (numeric - expected to be integer) 
-- PctImmigRecent: percentage of _immigrants_ who immigated within last 3 years (numeric - decimal) 
-- PctImmigRec5: percentage of _immigrants_ who immigated within last 5 years (numeric - decimal) 
-- PctImmigRec8: percentage of _immigrants_ who immigated within last 8 years (numeric - decimal) 
-- PctImmigRec10: percentage of _immigrants_ who immigated within last 10 years (numeric - decimal) 
-- PctRecentImmig: percent of _population_ who have immigrated within the last 3 years (numeric - decimal) 
-- PctRecImmig5: percent of _population_ who have immigrated within the last 5 years (numeric - decimal) 
-- PctRecImmig8: percent of _population_ who have immigrated within the last 8 years (numeric - decimal) 
-- PctRecImmig10: percent of _population_ who have immigrated within the last 10 years (numeric - decimal) 
-- PctSpeakEnglOnly: percent of people who speak only English (numeric - decimal) 
-- PctNotSpeakEnglWell: percent of people who do not speak English well (numeric - decimal) 
-- PctLargHouseFam: percent of family households that are large (6 or more) (numeric - decimal) 
-- PctLargHouseOccup: percent of all occupied households that are large (6 or more people) (numeric - decimal) 
-- PersPerOccupHous: mean persons per household (numeric - decimal) 
-- PersPerOwnOccHous: mean persons per owner occupied household (numeric - decimal) 
-- PersPerRentOccHous: mean persons per rental household (numeric - decimal) 
-- PctPersOwnOccup: percent of people in owner occupied households (numeric - decimal) 
-- PctPersDenseHous: percent of persons in dense housing (more than 1 person per room) (numeric - decimal) 
-- PctHousLess3BR: percent of housing units with less than 3 bedrooms (numeric - decimal) 
-- MedNumBR: median number of bedrooms (numeric - decimal) 
-- HousVacant: number of vacant households (numeric - expected to be integer) 
-- PctHousOccup: percent of housing occupied (numeric - decimal) 
-- PctHousOwnOcc: percent of households owner occupied (numeric - decimal) 
-- PctVacantBoarded: percent of vacant housing that is boarded up (numeric - decimal) 
-- PctVacMore6Mos: percent of vacant housing that has been vacant more than 6 months (numeric - decimal) 
-- MedYrHousBuilt: median year housing units built (numeric - may be integer) 
-- PctHousNoPhone: percent of occupied housing units without phone (in 1990, this was rare!) (numeric - decimal) 
-- PctWOFullPlumb: percent of housing without complete plumbing facilities (numeric - decimal) 
-- OwnOccLowQuart: owner occupied housing - lower quartile value (numeric - decimal) 
-- OwnOccMedVal: owner occupied housing - median value (numeric - decimal) 
-- OwnOccHiQuart: owner occupied housing - upper quartile value (numeric - decimal) 
-- OwnOccQrange: owner occupied housing - difference between upper quartile and lower quartile values (numeric - decimal) 
-- RentLowQ: rental housing - lower quartile rent (numeric - decimal) 
-- RentMedian: rental housing - median rent (Census variable H32B from file STF1A) (numeric - decimal) 
-- RentHighQ: rental housing - upper quartile rent (numeric - decimal) 
-- RentQrange: rental housing - difference between upper quartile and lower quartile rent (numeric - decimal) 
-- MedRent: median gross rent (Census variable H43A from file STF3A - includes utilities) (numeric - decimal) 
-- MedRentPctHousInc: median gross rent as a percentage of household income (numeric - decimal) 
-- MedOwnCostPctInc: median owners cost as a percentage of household income - for owners with a mortgage (numeric - decimal) 
-- MedOwnCostPctIncNoMtg: median owners cost as a percentage of household income - for owners without a mortgage (numeric - decimal) 
-- NumInShelters: number of people in homeless shelters (numeric - expected to be integer) 
-- NumStreet: number of homeless people counted in the street (numeric - expected to be integer) 
-- PctForeignBorn: percent of people foreign born (numeric - decimal) 
-- PctBornSameState: percent of people born in the same state as currently living (numeric - decimal) 
-- PctSameHouse85: percent of people living in the same house as in 1985 (5 years before) (numeric - decimal) 
-- PctSameCity85: percent of people living in the same city as in 1985 (5 years before) (numeric - decimal) 
-- PctSameState85: percent of people living in the same state as in 1985 (5 years before) (numeric - decimal) 
-- LemasSwornFT: number of sworn full time police officers (numeric - expected to be integer) 
-- LemasSwFTPerPop: sworn full time police officers per 100K population (numeric - decimal) 
-- LemasSwFTFieldOps: number of sworn full time police officers in field operations (on the street as opposed to administrative etc) (numeric - expected to be integer) 
-- LemasSwFTFieldPerPop: sworn full time police officers in field operations (on the street as opposed to administrative etc) per 100K population (numeric - decimal) 
-- LemasTotalReq: total requests for police (numeric - expected to be integer) 
-- LemasTotReqPerPop: total requests for police per 100K popuation (numeric - decimal) 
-- PolicReqPerOffic: total requests for police per police officer (numeric - decimal) 
-- PolicPerPop: police officers per 100K population (numeric - decimal) 
-- RacialMatchCommPol: a measure of the racial match between the community and the police force. High values indicate proportions in community and police force are similar (numeric - decimal) 
-- PctPolicWhite: percent of police that are caucasian (numeric - decimal) 
-- PctPolicBlack: percent of police that are african american (numeric - decimal) 
-- PctPolicHisp: percent of police that are hispanic (numeric - decimal) 
-- PctPolicAsian: percent of police that are asian (numeric - decimal) 
-- PctPolicMinor: percent of police that are minority of any kind (numeric - decimal) 
-- OfficAssgnDrugUnits: number of officers assigned to special drug units (numeric - expected to be integer) 
-- NumKindsDrugsSeiz: number of different kinds of drugs seized (numeric - expected to be integer) 
-- PolicAveOTWorked: police average overtime worked (numeric - decimal) 
-- LandArea: land area in square miles (numeric - decimal) 
-- PopDens: population density in persons per square mile (numeric - decimal) 
-- PctUsePubTrans: percent of people using public transit for commuting (numeric - decimal) 
-- PolicCars: number of police cars (numeric - expected to be integer) 
-- PolicOperBudg: police operating budget (numeric - may be integer) 
-- LemasPctPolicOnPatr: percent of sworn full time police officers on patrol (numeric - decimal) 
-- LemasGangUnitDeploy: gang unit deployed (numeric - integer - but really nominal - 0 means NO, 10 means YES, 5 means Part Time) 
-- LemasPctOfficDrugUn: percent of officers assigned to drug units (numeric - decimal) 
-- PolicBudgPerPop: police operating budget per population (numeric - decimal) 

-- murders: number of murders in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- murdPerPop: number of murders per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- rapes: number of rapes in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- rapesPerPop: number of rapes per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- robberies: number of robberies in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- robbbPerPop: number of robberies per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- assaults: number of assaults in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- assaultPerPop: number of assaults per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- burglaries: number of burglaries in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- burglPerPop: number of burglaries per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- larcenies: number of larcenies in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- larcPerPop: number of larcenies per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- autoTheft: number of auto thefts in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- autoTheftPerPop: number of auto thefts per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- arsons: number of arsons in 1995 (numeric - expected to be integer) potential GOAL attribute (to be predicted) 
-- arsonsPerPop: number of arsons per 100K population (numeric - decimal) potential GOAL attribute (to be predicted) 
-- ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal) GOAL attribute (to be predicted) 
-- nonViolPerPop: total number of non-violent crimes per 100K popuation (numeric - decimal) potential GOAL attribute (to be predicted) 
"""

def communityviolence_pd():
    columns = ['communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar', 'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ', 'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'ViolentCrimesPerPop', 'nonViolPerPop']
    table = []
    with open('/data/datasets/communityviolence.csv') as fin:
        for line in fin:
            line = line.strip().split(',')
            table.append(line)
    df = pd.DataFrame(table, columns=columns)
    df = df[df.ViolentCrimesPerPop != '?']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')
    return df


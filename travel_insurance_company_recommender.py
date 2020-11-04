import math
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

def data_preprocessing(X):
    date_columns = ['booking_date', 'dept_date', 'return_date', 'traveller_one_dob', 'traveller_two_dob', 'traveller_three_dob', 'traveller_four_dob']

    for column in date_columns:
        X[column] = pd.to_datetime(X[column], dayfirst=True)
            
    # Determine the booking window and drop the booking date
    X['booking_window'] = X['dept_date'] - X['booking_date']
    X['booking_window'] = X['booking_window'].apply(lambda length : length.days)

    X.drop('booking_date', axis=1, inplace=True)
    
    # Determine the travel length, then drop the departure_date and return_date
    X['travel_length'] = X['return_date'] - X['dept_date']
    X['travel_length'] = X['travel_length'].apply(lambda length : length.days)

    X.drop(['dept_date', 'return_date'], axis=1, inplace=True)
    
    now = datetime.today()

    # To convert dob to number of days lived
    dob_columns = ['traveller_one_dob', 'traveller_two_dob', 'traveller_three_dob', 'traveller_four_dob']

    for column in dob_columns:
        X[column] = X[column].apply(lambda dob : now - dob)
        X[column] = X[column].apply(lambda num_day_alive : num_day_alive.days)

    X['travel_one_days_alive'] = X['traveller_one_dob']
    X['travel_two_days_alive'] = X['traveller_two_dob']
    X['travel_three_days_alive'] = X['traveller_three_dob']
    X['travel_four_days_alive'] = X['traveller_four_dob']

    X.drop(['traveller_one_dob', 'traveller_two_dob', 'traveller_three_dob', 'traveller_four_dob'], axis=1, inplace=True)
    
    avg_days_alive = {
        'id': [],
        'avg_days_alive': []
    }

    for index, row in X.iterrows():
        total = 0
        if (not math.isnan(row['travel_one_days_alive'])):
            total += row['travel_one_days_alive']

        if (not math.isnan(row['travel_two_days_alive'])):
            total += row['travel_two_days_alive']

        if (not math.isnan(row['travel_three_days_alive'])):
            total += row['travel_three_days_alive']

        if (not math.isnan(row['travel_four_days_alive'])):
            total += row['travel_four_days_alive']

        avg = total/row['num_travellers']

        avg_days_alive['id'].append(row['id'])
        avg_days_alive['avg_days_alive'].append(avg)

    X = pd.merge(X, pd.DataFrame(avg_days_alive, columns=['id', 'avg_days_alive']), on='id')

    X.drop('travel_one_days_alive', axis=1, inplace=True)
    X.drop('travel_two_days_alive', axis=1, inplace=True)
    X.drop('travel_three_days_alive', axis=1, inplace=True)
    X.drop('travel_four_days_alive', axis=1, inplace=True)
    
    X.drop(columns="id", axis=1, inplace=True)

    cat_features = [f for f in X if X[f].dtype == 'object']
    
    for feature in cat_features:
        X = pd.concat((X, pd.get_dummies(X[feature], prefix=feature)), axis=1)
        del X[feature]   
    
    return X
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


    
    return X
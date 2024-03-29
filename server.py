from flask import Flask
from flask import request, jsonify, json

import pandas as pd

from travel_insurance_recommender import data_preprocessing
from travel_insurance_recommender import train_data

app = Flask(__name__)

# Train KNN
df = pd.read_csv("project_data.csv")

# For Company Recommendation
X1 = df.copy().drop(['ins_comp', 'subs_plan'], axis=1)
Y1 = df.copy()['ins_comp']

X1 =  data_preprocessing(X1)
knn_for_company = train_data(X1, Y1)

# For Plan Recommendation
X2 = df.copy().drop('subs_plan', axis=1)
Y2 = df.copy()['subs_plan']

X2 =  data_preprocessing(X2)
knn_for_plan = train_data(X2, Y2)

@app.route('/get_travel_insurance_company', methods=['GET', 'POST'])
def get_travel_insurance_company():
    # Use trained KNN to predict subsequent company.
    json_data = request.get_json()

    df = pd.read_csv("project_data.csv")
    incoming_data = pd.DataFrame(json_data, index=[0])
    incoming_data = df.append(incoming_data, ignore_index=True)

    X_test = incoming_data.copy().drop(['ins_comp', 'subs_plan'], axis=1)
    X_test = data_preprocessing(X_test)

    recommended_company = knn_for_company.predict(X_test[-1:])[0]
    ranking = knn_for_company.predict_proba(X_test[-1:])[0]

    print(recommended_company)
    print(ranking)

    # [ AIA AXA Allianze Aviva ]
    prob_aia = ranking[0]
    prob_axa = ranking[1]
    prob_allianze = ranking[2]
    prob_aviva = ranking[3]

    return jsonify({
        'recommended_company': recommended_company,
        'prob_aia': prob_aia,
        'prob_axa': prob_axa,
        'prob_allianze': prob_allianze,
        'prob_aviva': prob_aviva
    })

@app.route('/get_travel_insurance_plan', methods=['GET', 'POST'])
def get_travel_insurance_plan():
    # Use trained KNN to predict subsequent plan.
    json_data = request.get_json()

    df = pd.read_csv("project_data.csv")
    incoming_data = pd.DataFrame(json_data, index=[0])
    incoming_data = df.append(incoming_data, ignore_index=True)

    X_test = incoming_data.copy().drop('subs_plan', axis=1)
    X_test = data_preprocessing(X_test)

    ranking = knn_for_plan.predict_proba(X_test[-1:])[0]

    print(ranking)

    return jsonify({
        'prob_0': ranking[0],
        'prob_1': ranking[1],
        'prob_2': ranking[2]
    })

if __name__ == '__main__':
    app.run()
from flask import Flask
from flask import request, jsonify, json

import pandas as pd

# from travel_insurance_company_recommender import data_preprocessing
# from travel_insurance_company_recommender import train_data

app = Flask(__name__)

@app.route('/get_travel_insurance_company', methods=['GET', 'POST'])
def get_travel_insurance_company():


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


    ranking = knn_for_plan.predict_proba(X_test[-1:])[0]

    print(ranking)

    return jsonify({
        'prob_0': ranking[0],
        'prob_1': ranking[1],
        'prob_2': ranking[2]
    })

if __name__ == '__main__':
    app.run()

# Train KNN
df = pd.read_csv("project_data.csv")

X1 = df.copy().drop(['ins_comp', 'subs_plan'], axis=1)
Y1 = df.copy()['ins_comp']

X1 =  data_preprocessing(X1)
knn_for_company = train_data(X1, Y1)

X2 = df.copy().drop('subs_plan', axis=1)
Y2 = df.copy()['subs_plan']

X2 =  data_preprocessing(X2)
knn_for_plan = train_data(X2, Y2)
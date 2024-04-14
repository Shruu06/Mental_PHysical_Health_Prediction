# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:04:41 2021

"""

# %%

# importing libraries:
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# %%

app = Flask(__name__)

# %%
# jp note book req
model = pickle.load(open("models/ensemble.pkl", "rb"))

# %%


@app.route("/")
def home():
    return render_template("index.html")


# %%
Gender = {0: "Female", 1: "Male"}
family_history = {0: "No", 1: "Yes"}
work_interfere = {0: "Never", 1: "Often", 2: "Rarely", 3: "Sometimes"}
no_employees = {
    0: "05-Jan",
    1: "100-500",
    2: "25-Jun",
    3: "26-100",
    4: "500-1000",
    5: "More than 1000",
}
remote_work = {0: "No", 1: "Yes"}
tech_company = {0: "No", 1: "Yes"}
benefits = {0: "Don't know", 1: "No", 2: "Yes"}
care_options = {0: "No", 1: "Not sure", 2: "Yes"}
wellness_program = {0: "Don't know", 1: "No", 2: "Yes"}
seek_help = {0: "Don't know", 1: "No", 2: "Yes"}
anonymity = {0: "Don't know", 1: "No", 2: "Yes"}
leave = {
    0: "Don't know",
    1: "Somewhat difficult",
    2: "Somewhat easy",
    3: "Very difficult",
    4: "Very easy",
}
mental_health_consequence = {0: "Maybe", 1: "No", 2: "Yes"}
phys_health_consequence = {0: "Maybe", 1: "No", 2: "Yes"}
coworkers = {0: "No", 1: "Some of them", 2: "Yes"}
supervisor = {0: "No", 1: "Some of them", 2: "Yes"}
mental_health_interview = {0: "Maybe", 1: "No", 2: "Yes"}
phys_health_interview = {0: "Maybe", 1: "No", 2: "Yes"}
mental_vs_physical = {0: "Don't know", 1: "No", 2: "Yes"}
obs_consequence = {0: "No", 1: "Yes"}
treatment = {0: "No", 1: "Yes"}

# %%


@app.route("/mental_health", methods=["GET", "POST"])
def Individual():
    if request.method == "GET":
        return render_template("predict.html")

    if request.method == "POST":
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        data1 = request.form["Age"]
        data2 = request.form["Gender"]
        data3 = request.form["family_history"]
        data4 = request.form["work_interfere"]
        data5 = request.form["no_employees"]
        data6 = request.form["remote_work"]
        data7 = request.form["tech_company"]
        data8 = request.form["benefits"]
        data9 = request.form["care_options"]
        data10 = request.form["wellness_program"]
        data11 = request.form["seek_help"]
        data12 = request.form["anonymity"]
        data13 = request.form["leave"]
        data14 = request.form["mental_health_consequence"]
        data15 = request.form["phys_health_consequence"]
        data16 = request.form["coworkers"]
        data17 = request.form["supervisor"]
        data18 = request.form["mental_health_interview"]
        data19 = request.form["phys_health_interview"]
        data20 = request.form["mental_vs_physical"]
        data21 = request.form["obs_consequence"]

        # create original output dict
        output_dict = dict()
        output_dict["Age"] = data1
        output_dict["Gender"] = Gender[int(data2)]
        output_dict["family_history"] = family_history[int(data3)]
        output_dict["work_interfere"] = work_interfere[int(data4)]
        output_dict["no_employees"] = no_employees[int(data5)]
        output_dict["remote_work"] = remote_work[int(data6)]
        output_dict["tech_company"] = tech_company[int(data7)]
        output_dict["benefits"] = benefits[int(data8)]
        output_dict["care_options"] = care_options[int(data9)]
        output_dict["wellness_program"] = wellness_program[int(data10)]
        output_dict["seek_help"] = seek_help[int(data11)]
        output_dict["anonymity"] = anonymity[int(data12)]
        output_dict["leave"] = leave[int(data13)]
        output_dict["mental_health_consequence"] = mental_health_consequence[
            int(data14)
        ]
        output_dict["phys_health_consequence"] = phys_health_consequence[int(data15)]
        output_dict["coworkers"] = coworkers[int(data16)]
        output_dict["supervisor"] = supervisor[int(data17)]
        output_dict["mental_health_interview"] = mental_health_interview[int(data18)]
        output_dict["phys_health_interview"] = phys_health_interview[int(data19)]
        output_dict["mental_vs_physical"] = mental_vs_physical[int(data20)]
        output_dict["obs_consequence"] = obs_consequence[int(data21)]
        print(prediction, type(prediction))
        if prediction[0] == 1:
            result = "Sorry your mental health is not good.. Please consult a doctor."
        else:
            result = "Congragulations !!!! Your mental health is good..."
        return render_template("result.html", original_input=output_dict, result=result)


# %%

if __name__ == "__main__":
    app.run(debug=True)

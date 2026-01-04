import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np


def diabetic_duration(years):
    if years <= 5:
        return "Short"
    elif years <= 10:
        return "Medium"
    else:
        return "Long"

def age_group(age):
    if age <= 39:
        return "Young"
    elif age <= 55:
        return "Middle"
    else:
        return "Elderly"


def build_bn():

    model = DiscreteBayesianNetwork([
        ("CNN_DR", "Final_Risk"),
        ("Age_Group", "Final_Risk"),
        ("Diabetes_Duration", "Final_Risk")
    ])

    cpd_cnn = TabularCPD(
        variable="CNN_DR",
        variable_card=3,
        values=[[1/3], [1/3], [1/3]],
        state_names={"CNN_DR": ["Mild", "Moderate", "Severe"]}
    )

    cpd_age = TabularCPD(
        variable="Age_Group",
        variable_card=3,
        values=[[0.3], [0.4], [0.3]],
        state_names={"Age_Group": ["Young", "Middle", "Elderly"]}
    )

    cpd_duration = TabularCPD(
        variable="Diabetes_Duration",
        variable_card=3,
        values=[[0.4], [0.35], [0.25]],
        state_names={"Diabetes_Duration": ["Short", "Medium", "Long"]}
    )

    risk_values = np.zeros((3, 27))

    for i in range(27):
        if i < 9:          
            risk_values[:, i] = [0.7, 0.2, 0.1]
        elif i < 18:       
            risk_values[:, i] = [0.2, 0.6, 0.2]
        else:            
            risk_values[:, i] = [0.1, 0.2, 0.7]

    cpd_risk = TabularCPD(
        variable="Final_Risk",
        variable_card=3,
        values=risk_values,
        evidence=["CNN_DR", "Age_Group", "Diabetes_Duration"],
        evidence_card=[3, 3, 3],
        state_names={
            "Final_Risk": ["Low", "Medium", "High"],
            "CNN_DR": ["Mild", "Moderate", "Severe"],
            "Age_Group": ["Young", "Middle", "Elderly"],
            "Diabetes_Duration": ["Short", "Medium", "Long"]
        }
    )

    model.add_cpds(cpd_cnn, cpd_age, cpd_duration, cpd_risk)
    model.check_model()

    return VariableElimination(model)



def infer_risk(cnn_probs, age, duration_years):

    age_grp = age_group(age)
    duration = diabetic_duration(duration_years)

    states = ["Mild", "Moderate", "Severe"]

    cnn_probs = cnn_probs[:3]
    cnn_probs = cnn_probs / np.sum(cnn_probs)

    cnn_state = states[int(np.argmax(cnn_probs))]

    inference = build_bn()

    result = inference.query(
        variables=["Final_Risk"],
        evidence={
            "CNN_DR": cnn_state,
            "Age_Group": age_grp,
            "Diabetes_Duration": duration
        }
    )

    return result.values
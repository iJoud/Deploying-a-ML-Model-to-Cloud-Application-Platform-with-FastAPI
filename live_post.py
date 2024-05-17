import requests
import json


api = "https://deploying-a-ml-model-to-cloud-k5ve.onrender.com/predictions"

data = {"age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"}

respone = requests.post(api, json=data)

status_code = respone.status_code
pred = respone.json()

print("Response status code:\n", status_code)
print("Response content:\n", pred)

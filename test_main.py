from fastapi.testclient import TestClient
import json
from main import app


client = TestClient(app)


def test_greetings():
    request = client.get("/")

    assert request.status_code == 200
    assert request.json() == {"message:": "Welcome to my FastAPI app!"}


# rows
# 49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
# 52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K

def test_infrence0():

    data = json.dumps({
        "age": 49,
        "workclass": "Private",
        "fnlgt": 160187,
        "education": "9th",
        "education_num": 5,
        "marital_status": "Married-spouse-absent",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 16,
        "native_country": "Jamaica"
    })

    request = client.post("/predictions", data=data)

    print(request.json())

    assert request.status_code == 200
    assert request.json() == {'predictions': [' <=50K']}

# 52, Self-emp-inc, 287927, HS-grad, 9, Married-civ-spouse, Exec-managerial, Wife, White, Female, 15024, 0, 40, United-States, >50K

def test_infrence1():

    data = json.dumps({"age": 52,
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
                       "native_country": "United-States"})

    request = client.post("/predictions", data=data)

    assert request.status_code == 200
    assert request.json() == {'predictions': [' >50K']}

import uvicorn
from fastapi import FastAPI, Depends, Request
import pickle
# from model_development.churn_model import label_encoder, scaler
from models import ChurnPrediction, CreateUpdateChurnPredict
from database import engine, get_db, create_db_and_tables
import os
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
import  pandas as pd
import mlflow.pyfunc

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'
base_dir = os.getcwd()

# Modelin yüklenmesi
model_name = "ChurnModelPipeline"
model_version = 5
model_uri=f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()

create_db_and_tables()


def insert_churn(request, prediction, db):
    new_churn = ChurnPrediction(
        CreditScore=request["CreditScore"],
        Geography=request["Geography"],
        Gender=request["Gender"],
        Age=request["Age"],
        Tenure=request["Tenure"],
        Balance=request["Balance"],
        NumOfProducts=request["NumOfProducts"],
        HasCrCard=request["HasCrCard"],
        IsActiveMember=request["IsActiveMember"],
        EstimatedSalary=request["EstimatedSalary"],
        prediction=prediction
    )

    with db as session:
        session.add(new_churn)
        session.commit()
        session.refresh(new_churn)

    return new_churn


# Note that model is coming from mlflow
def make_churn_prediction(model, request):
    data_dict = dict(request)
    df = pd.DataFrame([data_dict])

    # Predict the output
    prediction = model.predict(df)

    return prediction[0]


# Advertising Prediction endpoint
@app.post("/prediction/churn")
async def predict_churn(request: CreateUpdateChurnPredict, fastapi_req: Request, db: Session = Depends(get_db)):
    prediction = make_churn_prediction(model, request.dict())

    db_insert_record = insert_churn(request=request.dict(), prediction=prediction,
                                    db=db)
    sonuc ="kalıcı"
    if prediction == 1:
        sonuc="gidici"

    return {"prediction": sonuc, "db_record": db_insert_record}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)

from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


# CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
# CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
#        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary
class ChurnPrediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: int
    prediction: float
    prediction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class CreateUpdateChurnPredict(SQLModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 690,
                "Geography": "France",
                "Gender": "Female",
                "Age": 29,
                "Tenure": 2,
                "Balance": 83807,
                "NumOfProducts": 86,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 6988
            }
        }

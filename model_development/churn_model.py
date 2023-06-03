from mlflow.tracking import MlflowClient
import os
import pandas as pd
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow.sklearn

# Çalışma dizini ve MLflow ayarları
base_dir = os.getcwd()
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

# Veri yükleme
data = pd.read_csv('/home/tolga/PycharmProjects/Week5/Odev/model_development/Churn_Modelling.csv')
data['Geography'] = data['Geography'].astype(str)
data['Gender'] = data['Gender'].astype(str)

# Encoder ve Scaler oluştur
one_hot_encoder = OneHotEncoder()
ordinal_encoder = OrdinalEncoder()
scaler = StandardScaler()

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Encoder ve Scaler işlemlerini kolonlara göre uygulamak için ColumnTransformer oluştur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', one_hot_encoder, ['Geography']),
        ('lab', ordinal_encoder, ['Gender'])
    ])

# Hedef ve özelliklerin ayırılması
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Eğitim ve test verilerinin ayırılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tahminci oluştur
number_of_trees = 10
estimator = RandomForestClassifier(n_estimators=number_of_trees)

# Bu dönüştürme işlemleri ve tahminciyi bir pipeline'da birleştir
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', estimator)])

# Pipeline'ı eğit
pipeline.fit(X_train, y_train)

# Tahminleme işlemi ve metriklerin hesaplanması
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# MLflow Experiment oluştur
experiment_name = "Churn_Experiment_Pipeline"
mlflow.set_experiment(experiment_name)
registered_model_name = "ChurnModelPipeline"

# Pipeline'ı MLflow ile kaydet
with mlflow.start_run(run_name="with-churn-rf-sklearn") as run:
    mlflow.log_param("n_estimators", number_of_trees)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=registered_model_name)


    else:
        mlflow.sklearn.log_model(estimator, "model")


# Optional part
name = registered_model_name
client = MlflowClient()

model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
print(model_uri)

mv = client.create_model_version(name, model_uri, run.info.run_id)
print("model version {} created".format(mv.version))
last_mv = mv.version
print(last_mv)
def print_models_info(models):
    for m in models:
        print("name: {}".format(m.name))
        print("latest version: {}".format(m.version))
        print("run_id: {}".format(m.run_id))
        print("current_stage: {}".format(m.current_stage))

def get_latest_model_version(models):
    for m in models:
        print("name: {}".format(m.name))
        print("latest version: {}".format(m.version))
        print("run_id: {}".format(m.run_id))
        print("current_stage: {}".format(m.current_stage))
    return m.version

models = client.get_latest_versions(name, stages=["None"])
print_models_info(models)

print(f"Latest version: { get_latest_model_version(models) }")



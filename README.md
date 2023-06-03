In this project we will deploy advertising model using mlflow model registry.

## 1. Start mlflow
```commandline
cd ~/02_mlops_docker

docker-compose up -d mysql mlflow minio
```

## 2. Copy/push your project to VM.
```commandline
mv 10_deploy_fastapi_with_mlflow_without_db/ 10
cd 10
```
## 3. Activate/create conda/virtual env
```commandline
conda activate fastapi

pip install -r requirements.txt
```

## 5. Train and register your model to mlflow
` python model_development/train_with_mlflow.py`


## 6. Learn your model version
- From  MLflow UI **learn model version**. Enter it main.py and copy/push main.py to VM.

## 7. Start uvicorn
```commandline
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```
## 8. Open docs
` http://localhost:8002/docs# `


## 9. Docker compose down
```commandline
cd ~/02_mlops_docker/; docker-compose down
```

\# 🚀 Industrial Equipment Failure Prediction for Predictive Maintenance



\---



\## 📌 Overview

Predictive maintenance helps prevent unexpected machine failures and reduces operational costs in industrial systems.



This project builds an \*\*end-to-end Machine Learning pipeline\*\* to predict whether an engine is likely to fail soon using sensor data from turbofan engines.



\---



\## 🎯 Objectives

\- Process raw sensor data  

\- Compute Remaining Useful Life (RUL)  

\- Convert to binary classification (failure vs healthy)  

\- Engineer meaningful features  

\- Train multiple ML models  

\- Select the best model using F1-score  

\- Deploy predictions using FastAPI  



\---



\## 📊 Dataset

We use the \*\*NASA CMAPSS FD001 dataset\*\*



🔗 Link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/



📁 Download the FD001 dataset and place it here:



```bash

data/raw/train\_FD001.txt



\## Project Structure

fsml\_project/

│── app/

│   ├── app.py

│   └── schema.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── logs/

│   └── app.log

│

├── models/

│   └── model\_v1.pkl

│

├── pipeline/

│   └── pipeline.py

│

├── src/

│   ├── data\_loader.py

│   ├── preprocess.py

│   ├── features.py

│   ├── train.py

│   ├── evaluate.py

│   ├── predict.py

│   └── utils.py

│

├── notebooks/

├── requirements.txt

├── Dockerfile

├── config.yaml

└── README.md



Preprocessing



from src.preprocess import preprocess\_pipeline



train\_df, val\_df, test\_df = preprocess\_pipeline("data/raw/train\_FD001.txt")



\##Feature Engineering



from src.features import add\_features



train\_df = add\_features(train\_df)

val\_df = add\_features(val\_df)

test\_df = add\_features(test\_df)



\##Model Training

| Model               | F1 Score |

| ------------------- | -------- |

| Logistic Regression | \~0.81    |

| Gradient Boosting   | \~0.84    |

| Random Forest ✅     | \~0.86    |



\##API Deployment (FastAPI)

uvicorn app.app:app --reload



\##Docker Setup

Build Image

docker build -t fsml\_project .

Run Container

docker run -p 8000:8000 fsml\_project



👉 Open: http://localhost:8000/docs



\##Full Pipeline



Run everything in one command:



python -m pipeline.pipeline

Pipeline Flow

Raw Data → Preprocessing → Feature Engineering → Training → Evaluation → Model Saving

📂 Outputs

Processed data → data/processed/

Model → models/model\_v1.pkl

Logs → logs/app.log




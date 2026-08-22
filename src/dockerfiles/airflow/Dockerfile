FROM apache/airflow:3.2.1

ENV PYTHONPATH=/opt/airflow

# CONTEXT 
WORKDIR /opt/airflow

COPY src/dockerfiles/airflow/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Code métier
COPY src/core /opt/airflow/core
COPY params.yml /opt/airflow/params.yml

# weather_training_scheduler_dag.py importe models.training_condition /
# models.training_state (logique de décision de déclenchement du training).
# On ne copie que ces deux fichiers, pas tout src/models/ : les autres
# scripts (grid_search, train_model, evaluate_model...) dépendent de
# lightgbm/mlflow/scikit-learn, volontairement absents de cette image —
# ils tournent dans le conteneur "models" via DockerOperator, pas ici.
COPY src/models/__init__.py /opt/airflow/models/__init__.py
COPY src/models/training_condition.py /opt/airflow/models/training_condition.py
COPY src/models/training_state.py /opt/airflow/models/training_state.py

import mlflow
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import cross_val_score


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize hyperparameters for a classifier using Optuna.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X_train : pandas.DataFrame
        Input features for training.
    y_train : pandas.Series
        Target variable for training.
    experiment_id : int
        ID of the MLflow experiment where results will be logged.

    Returns:
    --------
    float
        Mean F1 score of the classifier after cross-validation.
    """

    # Comienza el run de MLflow. Este run debería ser el hijo del run padre, 
    # así se anidan los diferentes experimentos.
    with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=f"Trial: {trial.number}", nested=True):

        # Parámetros a logguear
        params = {
            "objective": "clas:f1",
            "eval_metric": "f1"
        }

        # Sugiere valores para los hiperparámetros utilizando el objeto trial de optuna.
        classifier_name = trial.suggest_categorical('classifier', ['CatBoost', 
                                                                   'XGBoost'])
        
        if classifier_name == 'CatBoost':
            # CatBoost
            params["model"] = "CatBoost"

            iterations = trial.suggest_int('iterations', 100, 2000, step=100)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01)
            depth = trial.suggest_int('depth', 4, 10)

            params["iterations"] = iterations
            params["learning_rate"] = learning_rate
            params["depth"] = depth

            classifier_obj = cb.CatBoostClassifier(
                random_state=42, 
                logging_level='Silent', 
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth)
        else:
            # XGBoost
            params["model"] = "XGBoost"

            n_estimators = trial.suggest_int('n_estimators', 100, 1200, step=100)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01)
            max_depth = trial.suggest_int('max_depth', 3, 10)

            params["n_estimators"] = n_estimators
            params["learning_rate"] = learning_rate
            params["max_depth"] = max_depth

            classifier_obj = xgb.XGBClassifier(
                objective='binary:logistic', 
                random_state=42, 
                n_jobs=-1,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth)
        
        # Realizamos validación cruzada y calculamos el score F1
        score = cross_val_score(classifier_obj, X_train, y_train.to_numpy().ravel(), 
                                n_jobs=-1, cv=5, scoring='f1')
        
        # Log los hiperparámetros a MLflow
        mlflow.log_params(params)
        # Y el score f1 medio de la validación cruzada.
        mlflow.log_metric("f1", score.mean())

    return score.mean()

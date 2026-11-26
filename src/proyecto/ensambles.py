"""experiment005: bagging, votación y stacking sin reutilizar validación externa."""
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from .comun import cv, SEMILLA

def construir(bases, pipeline):
    """Bases preespecificadas: no usar hiperparámetros elegidos con toda la muestra.
    El stacking genera predicciones OOF dentro de cada fold externo.
    """
    estimadores = [(f"base{i}", pipeline(modelo, reducir))
                   for i, (modelo, _, reducir) in enumerate(bases.values())]
    bagging = pipeline(BaggingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=SEMILLA),
            n_estimators=20, max_samples=1.0, bootstrap=True,
            n_jobs=1, random_state=SEMILLA), False)
    meta = GridSearchCV(
        XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                      tree_method="hist", n_jobs=1, random_state=SEMILLA,
                      learning_rate=.1, verbosity=0),
        {"max_depth": [2, 3], "n_estimators": [50, 100]},
        cv=cv(), scoring="f1_macro", n_jobs=1)
    return {
        "Bagging (regresión logística)": bagging,
        "Votación (7 modelos)": VotingClassifier(estimators=estimadores, voting="hard", n_jobs=1),
        "Stacking (XGBoost)": StackingClassifier(
            estimators=estimadores, final_estimator=meta, cv=cv(),
            stack_method="auto", n_jobs=1, passthrough=False),
    }

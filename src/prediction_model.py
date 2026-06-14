import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#ker se ne ocenjuje natancnosti modela se temu nisem posebnno posvetil,
#malo sem raziskoval po internetu
#model seveda ne bo prevec natancen

FEATURE_COLUMNS = [
    "log_return",
    "abs_log_return",
    "squared_log_return",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_60d",
    "Volume"
]

#prvih 80% vzamemo za train data, ostalo za test natancnosti

def data_split(
        data: pd.DataFrame,
        train_size: float = 0.8
        ):
    
    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return train_data, test_data

#s tem naučimo model
def train_random_forest(train_data: pd.DataFrame):

    X_train = train_data[FEATURE_COLUMNS]
    Y_train = train_data['future_regime']


    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, Y_train)

    return model

#oceni model na testnih podatkih
def evaluate_model(model, 
                   test_data: pd.DataFrame):
    
    X_test = test_data[FEATURE_COLUMNS]
    Y_test = test_data['future_regime']

    prediction = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(Y_test, prediction))

    return prediction


#napovemo v kateri razred/ regime volatilnosti za zadnji razpoložljivi datum

def predict_latest(model,
                   data: pd.DataFrame):
    
    latest_row = data.iloc[[-1]]
    X_latest = latest_row[FEATURE_COLUMNS]

    predicted_class = model.predict(X_latest)[0]
    probabilities = model.predict_proba(X_latest)[0]
    classes = model.classes_

    probability_table = pd.DataFrame({
        'regime': classes,
        'probability' : probabilities
    }).sort_values('probability', ascending=False)

    return predicted_class, probability_table

def print_feature_importance(model):
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    print(importance)


from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def tune_random_forest(train_data: pd.DataFrame):
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["future_regime"]

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [20, 50],
        "min_samples_split": [10, 30],
        "max_features": [None]
    }

    time_split = TimeSeriesSplit(n_splits=5)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=time_split,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_   

    print("\nNajboljši parametri:")
    for param_name, param_value in best_params.items():
        print(f"{param_name}: {param_value}")


    print(f"\nNajboljši macro f1 score: {best_score:.4f}")

    return best_model, best_params, best_score
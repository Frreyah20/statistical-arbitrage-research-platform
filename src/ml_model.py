from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd 


def train_model(dataset):
    x = dataset.drop(columns = ["Pair", "Test Sharpe"])
    y = dataset["Test Sharpe"]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators = 100,random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    importance_df = pd.DataFrame({"Feature": x.columns, "Importance": model.feature_importances_})
    importance_df = importance_df.sort_values("Importance", ascending=False)
    print("\nFeature Importances:")
    print(importance_df)
    return model, score , importance_df


    
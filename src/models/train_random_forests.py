import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forests_model import RandomForestModel

if __name__ == "__main__":
    data_path = '../../data/processed/train_data.csv'
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestModel(random_state=42)
    model.train(X_train, y_train, n_estimators=59)
    
    # Evaluate the model on the testing data
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    model.save_model('random_forest_model.joblib')

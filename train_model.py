def build_models():
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor
    import joblib

    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']

    # Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X, y)
    joblib.dump(linear_regression, "linear_regression.joblib")

    # Logistic Regression
    y_categorical = (y > y.median()).astype(int)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X, y_categorical)
    joblib.dump(logistic_regression, "logistic_regression.joblib")

    # Decision Tree
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X, y)
    joblib.dump(decision_tree, "decision_tree.joblib")

build_models()

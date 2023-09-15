import numpy as np
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import yfinance as yf
def xgboostTick(tick):
    # %%



    # %%
    df = yf.download(tick, period='6mo', interval='1d')[:-1]
    df['Change'] = df['Adj Close'] - df['Open']
    df['Increased'] = (df['Change'] > 0).astype(bool)
    df['10-Day MA'] = df['Close'].rolling(window=5).mean()

    # scaler = MinMaxScaler()
    # df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    # %%
    # RSI

    # %%

    features = ['Open', 'High', 'Low', 'Volume', "10-Day MA"]
    X = df[features]
    y = df['Increased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # %%
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 10],
        'learning_rate': [0.01, 0.01, 0.1, 0.2],
        'subsample': [0.4, 0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    }

    model = XGBRFClassifier()
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10,
                                       scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # %%
    best_params = random_search.best_params_

    # %%
    final_model = XGBRFClassifier(**best_params)
    final_model.fit(X_train, y_train)
    prediction = final_model.predict(df[-1])['Increased']
    return prediction






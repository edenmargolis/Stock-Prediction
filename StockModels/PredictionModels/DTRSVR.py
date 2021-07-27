import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# DecisionTreeRegressor
class DTRSVR:

    # Compute mean absolute percentage error (MAPE)
    def get_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        print("true: ")
        print(y_true)
        print("pred: ")
        print(y_pred)
        print("y_true - y_pred : ")
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def DTRSVR_runner(self, mode, df):

        prices = df[df.columns[0:2]]
        prices["timestamp"] = pd.to_datetime(prices.date).astype(int) // (10 ** 9)
        dates = prices['date']
        prices = prices.drop(['date'], axis=1)

        dataset = prices.values
        X = dataset[:, 1].reshape(-1, 1)
        Y = dataset[:, 0:1]

        validation_size = 0.15
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                        random_state=seed)

        # Define model
        if mode == 0:
            model = DecisionTreeRegressor()
        else:
            model = SVR()

        # Fit to model
        model.fit(X_train, Y_train)
        # predict
        predictions = model.predict(X)
        print("Mape: --------- ")
        print(self.get_mape(Y, predictions))
        print("--------- ")

        fig = plt.figure(figsize=(24, 12))
        plt.xlabel("Date", fontsize='large')
        plt.ylabel("Price", fontsize='large')
        plt.plot(dates, Y)
        plt.plot(dates, predictions)
        plt.legend(["Original", "Predictions"])

        plt.show()
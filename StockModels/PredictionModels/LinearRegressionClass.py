import math
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class LinearRegressionClass:

    # Compute mean absolute percentage error (MAPE)
    def get_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ------------------------------------------------------------------------

    def get_preds_lin_reg(self, df, target_col, N, pred_min, offset):

        # Create linear regression object
        # fit_intercept - whether the data is expected to be centered, true is default.
        regr = LinearRegression(fit_intercept=True)

        # prediction list 
        pred_list = []

        for i in range(offset, len(df['adj_close'])):
            X_train = np.array(range(len(df['adj_close'][i - N:i])))  
            X_train = X_train.reshape(-1, 1)  
            # e.g X_train = [[0] [1] ...]

            y_train = np.array(df['adj_close'][i - N:i])  
            y_train = y_train.reshape(-1, 1)
            # e.g Y_train = [[721.89] [605.56] ...]

            # Train the model
            regr.fit(X_train, y_train)
            pred = regr.predict(np.array(N).reshape(1, -1))
            pred_list.append(pred[0][0])  # Predict the footfall using the model

        pred_list = np.array(pred_list)

        return pred_list

# ------------------------------------------------------------------------

    def calculate_linear_regression(self, debug, train_cv, Nmax, num_train, cv):
        RMSE = []
        R2 = []
        mape = []

        # N is no. of samples to use to predict the next value
        for N in range(1, Nmax + 1):
            est_list = self.get_preds_lin_reg(train_cv, 'adj_close', N, 0 , num_train)
            cv.loc[:, 'est' + '_N' + str(N)] = est_list
            RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
            R2.append(r2_score(cv['adj_close'], est_list))
            mape.append(self.get_mape(cv['adj_close'], est_list))
        if debug:
            print('RMSE = ' + str(RMSE))
            print('R2 = ' + str(R2))
            print('MAPE = ' + str(mape))
            
# ------------------------------------------------------------------------

    def LinearRegression_runner(self, debug, cv, N, train_cv, num_train):
        # argumants:
        # debug bit , train validation  , 
        self.calculate_linear_regression(debug, train_cv, N, num_train, cv)

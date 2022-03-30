# finalize model and make a prediction for monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

def df_to_supervised(data, target, n_in=1, n_out=1, dropnan=True):
    data[target] = data[target].shift(n_in)
    if dropnan:
        data.dropna(inplace=True)
    return data.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(model, train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, 1:], train[:, 0]
    # fit model
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

 #walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, 1:], test[i, 0]
        #print(test, testX, testy)
        # fit model on history and make a prediction
        yhat = xgboost_forecast(model, history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        #print('>expected=%.4f, predicted=%.4f' % (testy, yhat))
    # estimate prediction error
    error = sklearn.metrics.mean_absolute_error(test[:, 0], predictions)
    return error, test[:, 0], predictions

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("MAPE: ", mean_absolute_percentage_error(y_true=y_true_test,
                   y_pred=y_pred_test))


def ml_main(_df, _target):
    # transform the time series data into supervised learning
    data = df_to_supervised(_df, _target, n_in=3)
    #train, test = train_test_split(data, n_test)

    data1_index = int(0.7*data.shape[0])
    data1 = data[0:int(0.7*data.shape[0])]
    data2 = data[int(0.7*data.shape[0])+1:]



    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    # evaluate
    n_test = int(data1.shape[0]*0.01)
    print (data1.shape[0], n_test)

    mae, y, yhat = walk_forward_validation(data1, n_test)
    print('MAE: %.3f' % mae)
    train, test = train_test_split(data1, n_test)

    y_pred_test = model.predict(test[:,1:])
    y_true_test = test[:, 0]

    index_train, index_test = df_copy.index[0:data1.shape[0]-n_test], df_copy.index[data1.shape[0]-n_test:data1.shape[0]]


    y_pred_train = model.predict(train[:,1:])
    y_true_train = train[:,0]

    n_in = 3

    y_pred_extra = model.predict(data2[:,1:])
    print(y_pred_extra.shape, data2.shape)

    ##### dec = 4
    from sklearn.metrics import *
    print("mean_absolute_percentage_error: ", round(mean_absolute_percentage_error(y_true_test, y_pred_test), dec))
    print("max_error: ", round(max_error(y_true_test, y_pred_test), dec)) #max_error metric calculates the maximum residual error.
    print("mean_absolute_error: ", round(mean_absolute_error(y_true_test, y_pred_test), dec)) #Mean absolute error regression loss.
    print("mean_squared_error: ", round(mean_squared_error(y_true_test, y_pred_test), dec)) #Mean squared error regression loss.
    print("mean_squared_log_error: ", round(mean_squared_log_error(y_true_test, y_pred_test), dec)) #Mean squared logarithmic error regression loss.
    print("median_absolute_error: ", round(median_absolute_error(y_true_test, y_pred_test), dec)) #Median absolute error regression loss.
    #print("MAPE: ", metrics.mean_absolute_percentage_error(…)  #Mean absolute percentage error regression loss.
    print("r2_score: ", round(r2_score(y_true_test, y_pred_test), dec))   # (coefficient of determination) regression score function.
    print("mean_poisson_deviance: ", round(mean_poisson_deviance(y_true_test, y_pred_test), dec)) #Mean Poisson deviance regression loss.
    print("mean_gamma_deviance: ", round(mean_gamma_deviance(y_true_test, y_pred_test), dec)) #Mean Gamma deviance regression loss.
    print("mean_tweedie_deviance: ", round(mean_tweedie_deviance(y_true_test, y_pred_test), dec)) #Mean Tweedie deviance regression loss.
    #print("d2_tweedie_score: ", round(d2_tweedie_score(y_true_test, y_pred_test), dec))  # D^2  -*097653TWIOP]'.CXFJM;14 ion, percentage of Tweedie deviance explained.
    #print("mean_pinball_loss: ", round(mean_pinball_loss(y_true_test, y_pred_test), dec)) #Pinball loss for q/FVtile regression.


import pickle

def load_model(file_name):
    #"socorro_model_1.pkl"
    # load
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    # test
    ind = 1
    #test = X_val[ind]
    #xgb_model_loaded.predict(test)[0] == xgb_model.predict(test)[0]
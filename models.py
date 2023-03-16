import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


def read_and_process_file(csv):
    data = pd.read_csv(csv)
    data = data.rename(columns={"Unnamed: 0": "Date"})
    data.index = data["Date"]
    data = data.drop(columns="Date")
    dataT = data.T
    dataT["Date"] = dataT.index
    dataT["Date"] = pd.to_datetime(dataT["Date"][:-1])
    dataT.index = pd.to_datetime(dataT["Date"])
    return dataT


def plot_eda(df):
    plt.figure(figsize=(16, 8), dpi=150)
    df["Total"].plot(label="Total", color="red")
    plt.title("Number of referrals in Jan Plot")
    plt.xlabel("Jan to Feb Patients")
    plt.legend()
    plt.show()


def initialise_auto_arima(df):
    model = pm.auto_arima(
        df.Total,
        start_p=1,
        start_q=1,
        test="adf",  # use adftest to find optimal 'd'
        max_p=3,
        max_q=3,  # maximum p and q
        m=1,  # frequency of series
        d=None,  # let model determine 'd'
        seasonal=False,  # No Seasonality
        start_P=0,
        D=0,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def initialise_arima(df):
    model = pm.auto_arima(
        df.Total,
        start_p=1,
        start_q=1,
        test="adf",  # use adftest to find optimal 'd'
        max_p=3,
        max_q=3,  # maximum p and q
        m=1,  # frequency of series
        d=None,  # let model determine 'd'
        seasonal=False,  # No Seasonality
        start_P=0,
        D=0,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    train = df.Total[:30]
    test = df.Total[30:]

    model = ARIMA(train, order=(2, 1, 0))
    model_fit = model.fit()
    arr = model_fit.predict(start=30, end=50, alpha=0.1, dynamic=True)  # 95% conf
    arr = round(arr)

    arr.index = pd.to_datetime(arr.index)
    plt.figure(figsize=(40, 5), dpi=100)
    plt.plot(df.Total, label="training")
    plt.plot(arr, label="forecast")
    plt.title("Forecast vs Actuals")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()

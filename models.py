import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import streamlit as st


def read_and_process_file(csv):
    data = pd.read_csv(csv)
    data = data.rename(columns={"Unnamed: 0": "Date"})
    data.index = data["Date"]
    data = data.drop(columns="Date")
    dataT = data.T
    dataT["Date"] = dataT.index
    dataT["Date"] = pd.to_datetime(dataT["Date"])
    dataT.index = pd.to_datetime(dataT["Date"])
    return dataT


def plot_eda(df):
    plt.figure(figsize=(16, 8), dpi=150)
    df["Total"].plot(label="Total", color="red")
    plt.title("Number of referrals in Jan Plot")
    plt.xlabel("Jan to Feb Patients")
    plt.legend()
    plt.show()


def initialise_arima_prediction(df):
    totallen = df.shape[0]
    predtime = totallen - 1
    daystopred = 15

    train = df.Total[:predtime]

    model = ARIMA(train, order=(2, 0, 2))
    model_fit = model.fit()
    arr = model_fit.predict(
        start=predtime, end=predtime + daystopred, alpha=0.1, dynamic=True
    )  # 95% conf

    arr = round(arr)
    arr.index = pd.date_range(df.index[predtime], periods=daystopred + 1, freq="D")
    arr.index = pd.to_datetime(arr.index)

    plt.figure(figsize=(20, 5), dpi=100)
    plt.plot(df.Total[predtime - 15 :], label="training")
    plt.plot(arr, label="forecast")

    plt.title("Forecast vs Actuals")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()
    st.session_state.estimated_time = arr


def estimated_time_prediction(df):
    ##Estimating time
    newestimates = pd.DataFrame(df)
    newestimates = newestimates.rename(
        columns={"predicted_mean": "Estimate number of patient"}
    )
    # From excel data we find that Cogitive test (MMSE) consists of 33.92% of referral and Montreal Cognitive Assessment consists of 32.74% of referrals rest is variance
    newestimates["EstimatedTime"] = (
        newestimates["Estimate number of patient"]
        * (0.3392 * 15 + 0.3274 * 20 + 5 / 3)
        / 60
    )
    newestimates = newestimates.drop(columns=["Estimate number of patient"])

    fig, ax = plt.subplots(figsize=(15, 6))
    newestimates.plot(kind="bar", ax=ax)
    plt.title("Estimated Time")
    plt.show()

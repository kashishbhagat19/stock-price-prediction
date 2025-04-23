import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model 
import streamlit as st 
from datetime import date

import yfinance as yf
# start = '2010-01-01'
# end = '2026-12-31'

#New
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction App")
st.caption("Powered by LSTM, TensorFlow, and yFinance")
st.markdown("---")
#End

st.title('Stock Price Prediction')

# user_input = st.text_input('Enter Stock Ticker','AAPL')
#New
popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
user_input = st.selectbox("Choose a Stock Ticker", popular_tickers, index=0)
custom_ticker = st.text_input("Or enter custom ticker (e.g., NFLX)", "")
if custom_ticker:
    user_input = custom_ticker.upper()
st.markdown("### 📍 Selected Stock Ticker: " + user_input)
st.markdown(f"[🔎 View {user_input} on Yahoo Finance](https://finance.yahoo.com/quote/{user_input})", unsafe_allow_html=True)

try:
    ticker_info = yf.Ticker(user_input).info
    company_name = ticker_info.get("longName", "N/A")
    sector = ticker_info.get("sector", "N/A")
    country = ticker_info.get("country", "N/A")
    summary = ticker_info.get("longBusinessSummary", "Summary not available.")

    st.markdown(f"### 🏢 {company_name}")
    st.markdown(f"**Sector:** {sector} | **Country:** {country}")
    st.markdown(f"**🔍 Company Overview:**\n\n{summary}")
except Exception as e:
    st.warning("Could not fetch company info. Check if the ticker is valid.")
#End

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date')
with col2:
    end_date = st.date_input('End Date')

if start_date >= end_date:
    st.error("⚠️ End date must be after start date.")
else:
    with st.spinner("📡 Downloading data and predicting..."):#New
        df = yf.download(user_input, start = start_date, end = end_date)


#Describing the Data
        if not df.empty:
            st.subheader(f'Data from {start_date.year} - {end_date.year}')
            st.write(df.describe())
        else:
            st.warning("No data found. Check your stock symbol or date range.")



#Visualization
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize =(12,6))
        plt.plot(df.Close)
        st.pyplot(fig)
    st.success("✅ Prediction Complete!")#New

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

#Splitting data into Training and Testing

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)


#Load my model
    model = load_model('keras_model.h5')


#Testing Part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test,y_test = np.array(x_test),np.array(y_test)

    y_predicted = model.predict(x_test)
    # scale_values = scaler.scale_#scalar =scale_values

    # scale_factor = 1/scale_values[0]
    # y_predicted = y_predicted * scale_factor
    # y_test = y_test * scale_factor

    # #New
    # # Show latest predicted closing price
    # st.subheader("📊 Latest Predicted Closing Price")
    # latest_price = y_predicted[-1]
    # actual_price = y_test[-1]
    # price_diff = latest_price - actual_price
    # change = "🔺" if price_diff > 0 else "🔻"
    # st.metric("Predicted Price", f"${latest_price:.2f}")
    # st.metric("Price Difference", f"{change}${abs(price_diff):.2f}")
    # #End
    scale_values = scaler.scale_
    scale_factor = 1 / scale_values[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    if len(y_predicted) > 0 and len(y_test) > 0:
        latest_price = float(y_predicted[-1])
        actual_price = float(y_test[-1])
        price_diff = latest_price - actual_price
        change = "🔺" if price_diff > 0 else "🔻"
        st.subheader("📊 Latest Predicted Closing Price")
        st.metric("Predicted Price", f"${latest_price:.2f}")
        st.metric("Price Difference", f"{change}${abs(price_diff):.2f}")
    else:
        st.warning("Prediction data is empty. Check your date range or model input.")
# End

#Final Graph

    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label = 'Original Price')
    plt.plot(y_predicted,'r',label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
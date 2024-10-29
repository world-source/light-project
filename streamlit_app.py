import streamlit as st 
from datetime import date
import numpy as np
np.float_ = np.float64
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import pandas as pd

st.set_page_config(
    page_title="Light Project",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdbM0Bqr7Q7mCAouhY1p_x_poXPrxinl9a7Q&s",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Light project is a machine learning algorithm written by students of Geelong High School to predict stock prices for American stock data."
    }
)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown("<h1 style='text-align: center; color: black;'>The light ahead.</h1>", unsafe_allow_html=True)

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA', 'NVDA', 'INTC', 'AMZN', 'EBAY', 'AAL', 'AMD', 'NFLX', 'PEP', 'ADBE', 'META', 'TXN', 'ABNB', 'PYPL', 'LYFT')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']].copy()

# Verify if 'Close' column exists
if 'Close' not in df_train.columns:
    st.error("Error: The 'Close' column is not found in the dataset.")
    st.stop()

# Rename columns and clean data
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Check what df_train looks like after renaming
st.write("DataFrame after renaming columns:")
st.write(df_train)

# Check if 'ds' and 'y' columns exist after renaming
if 'ds' not in df_train.columns or 'y' not in df_train.columns:
    st.error("Error: Renaming columns failed.")
    st.stop()

# Check if 'y' is a Series
if not isinstance(df_train['y'], pd.Series):
    st.error("Error: Column 'y' is not a Series.")
    st.stop()

# Convert 'y' to numeric values, coercing errors
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

# Check for NaN values in 'y'
st.write("Number of NaN values in 'y':", df_train['y'].isna().sum())

# Drop rows with NaN values in 'y'
df_train = df_train.dropna(subset=['y'])

# Ensure that there are enough rows to fit the model
if df_train.shape[0] < 2:
    st.error("Error: Not enough valid rows after cleaning the data.")
    st.stop()

# Print columns of df_train for verification
st.write("Columns in training data:", df_train.columns)
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

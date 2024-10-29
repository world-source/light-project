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

# Prepare the training DataFrame for Prophet
df_train = data[['Date', 'Close']]

# Rename columns
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Check DataFrame before conversion
st.write("DataFrame before conversion:")
st.write(df_train)

# Check for the expected columns
if 'y' not in df_train.columns:
    st.error("Error: 'y' column is missing from the DataFrame.")
    st.stop()

# Check the type of 'y' and its contents
st.write("Contents of 'y' before conversion:")
st.write(df_train['y'])
st.write("Type of 'y':", type(df_train['y']))

# Ensure 'ds' is a datetime type and 'y' is numeric
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')  # Ensure ds is datetime

# Check if 'y' is a list or array-like before conversion
if isinstance(df_train['y'], (pd.Series, list, np.ndarray)):
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert y to numeric
else:
    st.error("Error: 'y' is not a valid list, Series, or array-like.")
    st.stop()

# Check for NaN values in 'y'
if df_train['y'].isnull().any():
    st.warning("Warning: There are NaN values in the 'y' column after conversion.")

# Display the DataFrame after conversion
st.write("DataFrame after conversion:")
st.write(df_train)

# Drop rows with NaN values in 'y'
df_train = df_train.dropna(subset=['y'])

# Check if df_train has enough data
if df_train.shape[0] < 2:
    st.error("Error: Not enough valid rows after cleaning the data.")
    st.stop()

# Verify 'y' is a Series
if not isinstance(df_train['y'], pd.Series):
    st.error("Error: Column 'y' is not a Series.")
    st.stop()

# Fit the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dataframe and make predictions
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

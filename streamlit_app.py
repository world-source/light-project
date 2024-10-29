import streamlit as st 
from datetime import date
import numpy as np
import pandas as pd  # Ensure you import pandas
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

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

# Prepare the training DataFrame for Prophet
df_train = data[['Date', 'Close']]

# Rename columns
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Check DataFrame structure
st.write("DataFrame before conversion:")
st.write(df_train)
st.write("Data types of DataFrame columns:")
st.write(df_train.dtypes)

# Check if 'y' column is present and check its type
if 'y' not in df_train.columns:
    st.error("Error: 'y' column is missing from the DataFrame.")
    st.stop()

# Check the type and contents of 'y'
st.write("Contents of 'y' before conversion:")
st.write(df_train['y'])
st.write("Type of 'y':", type(df_train['y']))

# Ensure 'y' is numeric and check for NaN values
try:
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert y to numeric
except Exception as e:
    st.error(f"Error during conversion: {e}")
    st.stop()

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
Key Changes

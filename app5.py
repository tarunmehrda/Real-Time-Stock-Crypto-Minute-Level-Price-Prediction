import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import Callback
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
import pytz

# ======================
# --- Custom Callback for Streamlit ---
# ======================
class StreamlitProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training Progress: Epoch {epoch+1}/{self.epochs} - Loss: {logs['loss']:.6f}")

# ======================
# --- Functions ---
# ======================
def fetch_latest_minute_data(ticker, lookback_days=1):
    # Handle custom ticker input
    if ticker.upper() in ['NIFTY', 'NIFTY 50', 'NIFTY50']:
        ticker = '^NSEI'
    elif ticker.upper() in ['SENSEX', 'BSE SENSEX']:
        ticker = '^BSESN'
    elif ticker.upper() in ['BTCUSD', 'BTC']:
        ticker = 'BTC-USD'
    elif ticker.upper() in ['ETHUSD', 'ETH']:
        ticker = 'ETH-USD'
    elif not ticker.endswith(('.NS', '.BO', '.NYSE', '.NASDAQ')) and not ticker.startswith('^') and '.' not in ticker:
        # Only add .NS suffix for Indian stocks if not already present and it's not a crypto or other asset
        # Don't add .NS to crypto assets like BTC-USD, ETH-USD, etc.
        if ticker.upper() not in ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT']:
            ticker = f"{ticker}.NS"
    
    try:
        stock_data = yf.download(ticker, period=f"{lookback_days}d", interval="1m")
        if stock_data.empty:
            st.error(f"No minute-level data available for {ticker}. Try: BTC-USD, ^NSEI (Nifty), ^BSESN (Sensex), RELIANCE, TCS")
            return None, None
        
        # Convert to Indian Standard Time
        ist = pytz.timezone('Asia/Kolkata')
        if stock_data.index.tz is None:
            stock_data.index = stock_data.index.tz_localize('UTC')
        stock_data.index = stock_data.index.tz_convert(ist)
        
        return stock_data['Close'].values.reshape(-1, 1), stock_data.index
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def prepare_data(data, time_steps):
    # Check if data is sufficient for the time_steps
    if len(data) <= time_steps:
        st.error(f"Insufficient data for the selected time steps. Data length: {len(data)}, Time steps: {time_steps}")
        return np.array([]), np.array([]), None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    
    # Check if we have any data
    if len(X) == 0 or len(y) == 0:
        st.error("No training data generated. Please check your data source and parameters.")
        return np.array([]), np.array([]), None
        
    return np.array(X), np.array(y), scaler

def create_model(time_steps):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.3),
        LSTM(units=100, return_sequences=True),
        Dropout(0.3),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def generate_future_dates(last_date, num_minutes):
    ist = pytz.timezone('Asia/Kolkata')
    if last_date.tzinfo is None:
        last_date = last_date.tz_localize('UTC')
    last_date = last_date.tz_convert(ist)
    future_dates = [last_date + timedelta(minutes=i+1) for i in range(num_minutes)]
    return pd.DatetimeIndex(future_dates)

def predict_future(model, last_sequence, scaler, n_future):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(n_future):
        current_prediction = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        future_predictions.append(current_prediction[0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_prediction
    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

def plot_predictions_with_future(dates, actual_values, train_predictions, test_predictions, 
                                 future_dates, future_predictions, train_size, time_steps, ticker):
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_dates = dates[time_steps:]
    
    ax.plot(plot_dates, actual_values.flatten(), label='Actual Prices', color='blue', linewidth=2)
    ax.plot(plot_dates[:train_size], train_predictions.flatten(), label='Training Predictions', color='green', linestyle='--', linewidth=2)
    ax.plot(plot_dates[train_size:], test_predictions.flatten(), label='Testing Predictions', color='red', linestyle='--', linewidth=2)
    ax.plot(future_dates, future_predictions.flatten(), label='Future Predictions', color='purple', linestyle='--', linewidth=2, marker='o', markersize=3)
    
    ax.axvline(x=plot_dates[train_size], color='gray', linestyle='-', label='Train/Test Split')
    ax.axvline(x=plot_dates[-1], color='gray', linestyle=':', label='Prediction Start')
    
    future_std = np.std(test_predictions - actual_values[train_size:])
    ax.fill_between(future_dates,
                     (future_predictions - future_std * 2).flatten(),
                     (future_predictions + future_std * 2).flatten(),
                     color='purple', alpha=0.1, label='95% Confidence Interval')
    
    ax.set_title(f'{ticker} Minute-Level Prediction (IST)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (IST)', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, rmse, mae, mape

# ======================
# --- Streamlit App ---
# ======================
st.title("📈 Stock & Crypto Minute-Level Price Prediction (LSTM)")
st.markdown("**Time Zone: Indian Standard Time (IST)**")

st.sidebar.header("Configuration")

# Ticker selection with custom input
ticker_option = st.sidebar.selectbox(
    "Choose ticker type:",
    ["Popular Assets", "Custom Ticker"]
)

if ticker_option == "Popular Assets":
    popular_tickers = {
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "Nifty 50 (^NSEI)": "^NSEI",
        "Sensex (^BSESN)": "^BSESN",
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS",
        "Infosys (INFY.NS)": "INFY.NS",
        "Apple (AAPL)": "AAPL",
        "Tesla (TSLA)": "TSLA",
        "Google (GOOGL)": "GOOGL"
    }
    selected_ticker = st.sidebar.selectbox("Select Asset:", list(popular_tickers.keys()))
    ticker = popular_tickers[selected_ticker]
else:
    ticker = st.sidebar.text_input("Enter Custom Ticker:", "BTC-USD")
    st.sidebar.info("💡 Examples: BTC-USD, ETH-USD, ^NSEI, ^BSESN, RELIANCE, TCS, AAPL, TSLA")

# Model parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")

lookback_days = st.sidebar.slider("Lookback Days", 1, 7, 2)
time_steps = st.sidebar.slider("Time Steps (LSTM)", 10, 120, 60)
train_split = st.sidebar.slider("Training Split (%)", 50, 90, 80)/100
future_minutes = st.sidebar.slider("Future Prediction Minutes", 5, 60, 30)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 30)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)

# Additional custom settings
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Settings")

show_raw_data = st.sidebar.checkbox("Show Raw Data", False)
show_model_summary = st.sidebar.checkbox("Show Model Summary", False)

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Ticker Help")
st.sidebar.info("""
**Crypto:**
- Bitcoin: BTC-USD
- Ethereum: ETH-USD

**Indian Indices:**
- Nifty 50: ^NSEI
- Sensex: ^BSESN

**Indian Stocks:**
- Reliance: RELIANCE
- TCS: TCS
- Infosys: INFY

**US Stocks:**
- Apple: AAPL
- Tesla: TSLA
- Google: GOOGL
""")

# Main app
st.write(f"Fetching minute-level data for **{ticker}** over the last {lookback_days} day(s)...")

data, dates = fetch_latest_minute_data(ticker, lookback_days)

if data is not None:
    st.success(f"✅ Fetched {len(data)} data points in IST")
    
    # Show raw data if requested
    if show_raw_data:
        st.subheader("📋 Raw Data Preview")
        raw_df = pd.DataFrame({
            'Timestamp': dates,
            'Close Price': data.flatten()
        })
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Price", f"${data[0][0]:.2f}")
        with col2:
            st.metric("Last Price", f"${data[-1][0]:.2f}")
        with col3:
            price_change = data[-1][0] - data[0][0]
            st.metric("Total Change", f"${price_change:+.2f}")
        with col4:
            change_pct = (price_change / data[0][0]) * 100
            st.metric("Change %", f"{change_pct:+.2f}%")
    
    X, y, scaler = prepare_data(data, time_steps)
    
    # Check if we have sufficient data
    if X.size == 0 or y.size == 0:
        st.error("No training data available. Please adjust parameters and try again.")
    elif len(X) < 2:
        st.error("Insufficient training data. Please increase lookback days or decrease time steps.")
    else:
        train_size = int(len(X) * train_split)
        
        # Ensure we have at least some data for training and testing
        if train_size == 0 or train_size >= len(X):
            st.error("Training data split resulted in insufficient samples. Please adjust training split percentage.")
        else:
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Ensure we have data for both training and testing
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("Data split resulted in empty training or testing sets. Please adjust parameters.")
            else:
                st.write(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                
                # Show model summary if requested
                if show_model_summary:
                    st.subheader("🧠 Model Architecture")
                    model = create_model(time_steps)
                    # Capture model summary
                    summary_list = []
                    model.summary(print_fn=lambda x: summary_list.append(x))
                    st.text("\n".join(summary_list))
                else:
                    model = create_model(time_steps)
                
                st.write("Training LSTM model...")
                callback = StreamlitProgressCallback(epochs)
                
                # Add validation split only if we have enough data
                if len(X_train) > 10:
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                             validation_split=0.1, verbose=0, callbacks=[callback])
                else:
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                             verbose=0, callbacks=[callback])
                
                train_predictions = model.predict(X_train)
                test_predictions = model.predict(X_test)
                
                last_sequence = X[-1]
                future_predictions = predict_future(model, last_sequence, scaler, future_minutes)
                future_dates = generate_future_dates(dates[-1], future_minutes)
                
                train_predictions = scaler.inverse_transform(train_predictions)
                test_predictions = scaler.inverse_transform(test_predictions)
                actual_values = scaler.inverse_transform(y)
                
                # Calculate accuracy metrics
                train_mse, train_rmse, train_mae, train_mape = calculate_metrics(
                    actual_values[:train_size], train_predictions)
                test_mse, test_rmse, test_mae, test_mape = calculate_metrics(
                    actual_values[train_size:], test_predictions)
                
                # Display metrics
                st.subheader("📊 Model Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training RMSE", f"${train_rmse:.2f}")
                    st.metric("Training MAE", f"${train_mae:.2f}")
                    st.metric("Training MAPE", f"{train_mape:.2f}%")
                with col2:
                    st.metric("Testing RMSE", f"${test_rmse:.2f}")
                    st.metric("Testing MAE", f"${test_mae:.2f}")
                    st.metric("Testing MAPE", f"{test_mape:.2f}%")
                
                st.subheader("📈 Prediction Chart")
                plot_predictions_with_future(dates, actual_values, train_predictions, test_predictions,
                                             future_dates, future_predictions, train_size, time_steps, ticker)
                
                st.subheader("⏱️ Future Predictions (IST)")
                prediction_table = pd.DataFrame({
                    'Time (IST)': future_dates.strftime('%Y-%m-%d %H:%M:%S'),
                    'Predicted Price': future_predictions.flatten()
                })
                st.dataframe(prediction_table.style.format({"Predicted Price": "${:,.2f}"}))
                
                # Display current and predicted price change
                current_price = actual_values[-1][0]
                future_price = future_predictions[-1][0]
                price_change = future_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                st.subheader("💰 Price Forecast Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:,.2f}")
                with col2:
                    st.metric(f"Predicted Price ({future_minutes}min)", f"${future_price:,.2f}", 
                             f"{price_change:+,.2f}")
                with col3:
                    st.metric("Expected Change", f"{price_change_pct:+.2f}%")
                
                # Download predictions
                st.subheader("💾 Download Predictions")
                csv = prediction_table.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"{ticker}_predictions.csv",
                    mime="text/csv"
                )
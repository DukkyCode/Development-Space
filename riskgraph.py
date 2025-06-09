# risk_metric.py
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def fetch_btc_data() -> pd.DataFrame:
    # Download historical daily data
    df = yf.download(tickers='BTC-USD', start='2010-07-01', interval='1d', auto_adjust=False)

    # Flatten multi-index if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Check for 'Close' in columns
    if 'Close' not in df.columns:
        raise ValueError("Missing 'Close' column in downloaded daily data")

    df.reset_index(inplace=True)
    df = df[['Date', 'Close']].rename(columns={'Close': 'Value'})
    df = df[df['Value'] > 0].sort_values('Date').copy()

    # Download recent minute-level data
    live = yf.download(tickers='BTC-USD', period='1d', interval='1m', auto_adjust=False)
    if isinstance(live.columns, pd.MultiIndex):
        live.columns = live.columns.get_level_values(0)

    if not live.empty and 'Close' in live.columns:
        latest_price = live['Close'].iloc[-1].item()
        df.loc[len(df)] = [pd.Timestamp(date.today()), latest_price]

    return df.reset_index(drop=True)

def compute_risk_metric(df: pd.DataFrame, moving_average_days=365, diminishing_factor=0.395):
    df['MA'] = df['Value'].rolling(moving_average_days, min_periods=1).mean()
    df = df[df['MA'].notna()].copy().reset_index(drop=True)

    index_factors = np.power(np.arange(len(df), dtype=float), diminishing_factor)
    log_returns = np.log(df['Value'].astype(float).to_numpy().flatten()) - \
                  np.log(df['MA'].astype(float).to_numpy().flatten())
    df['Preavg'] = log_returns * index_factors
    df['avg'] = (df['Preavg'] - df['Preavg'].min()) / (df['Preavg'].max() - df['Preavg'].min())
    return df

def predict_price_per_risk(df: pd.DataFrame, diminishing_factor=0.395):
    max_preavg = df['Preavg'].max()
    min_preavg = df['Preavg'].min()
    current_index = len(df) - 1
    log_ma = np.log(df['MA'].iloc[-1])

    return {
        round(risk, 1): round(np.exp(
            (risk * (max_preavg - min_preavg) + min_preavg) /
            (current_index ** diminishing_factor) + log_ma
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

def plot_price_and_risk(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], name='Price (USD)', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'], name='Risk (0-1)', line=dict(color='white')), secondary_y=True)

    # Buy zone (green)
    for i in range(5, 0, -1):
        fig.add_hrect(y0=i*0.1, y1=(i-1)*0.1, fillcolor='green', opacity=0.2 + 0.05*i, line_width=0, secondary_y=True)
    # Sell zone (red)
    for i in range(6, 10):
        fig.add_hrect(y0=i*0.1, y1=(i+1)*0.1, fillcolor='red', opacity=0.1*i - 0.4, line_width=0, secondary_y=True)

    fig.update_layout(
        template='plotly_dark',
        title=f"Price vs. Risk — Updated {df['Date'].iloc[-1].strftime('%Y-%m-%d')} | "
              f"Price: ${round(df['Value'].iloc[-1])} | Risk: {round(df['avg'].iloc[-1], 2)}"
    )
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', secondary_y=False)
    fig.update_yaxes(title='Risk (0–1)', range=[0, 1], dtick=0.1, secondary_y=True)
    return fig

def plot_colored_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x='Date', y='Value', color='avg', color_continuous_scale='jet')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_layout(template='plotly_dark', title='BTC Price Colored by Risk Metric')
    return fig

def generate_all_risk_plots():
    df = fetch_btc_data()
    df = compute_risk_metric(df)
    price_table = predict_price_per_risk(df)

    price_vs_risk = plot_price_and_risk(df)
    risk_scatter = plot_colored_scatter(df)

    return {
        "price_vs_risk": price_vs_risk,
        "risk_scatter": risk_scatter,
    }


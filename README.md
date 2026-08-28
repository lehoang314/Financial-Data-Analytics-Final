# Financial-Data-Analytics-Final
Assignment Overview
1. Download 10 daily stocks and index data (closing prices and index). 
2. Apply the Augmented Dickey-Fuller (ADF) test to all time series. 
3. Regress each stock's return on the market index return.
4. Conduct a panel data regression with all selected stocks.
5. Build a modular system using object-oriented programming (OOP) to perform the above analyses.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare

# --- Part B: Time Series Analysis ---
class TimeSeriesAnalyzer:
    def __init__(self, series, name):
        self.series = series.dropna()
        self.name = name

    def plot_series(self):
        self.series.plot(title=f"Time Series: {self.name}", figsize=(10, 4))
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(f"{self.name}_timeseries.png")
        plt.close()

    def adf_test(self):
        result = adfuller(self.series)
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Stationary': result[1] < 0.05
        }

    def difference_and_test(self):
        differenced = self.series.diff().dropna()
        result = adfuller(differenced)
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Stationary': result[1] < 0.05
        }

# --- Part C: Return Regression ---
class ReturnRegressor:
    def compute_returns(self, price_series):
        return np.log(price_series / price_series.shift(1)).dropna()

    def run_regression(self, dependent, independent):
        df = pd.concat([dependent, independent], axis=1).dropna()
        X = sm.add_constant(df.iloc[:, 1])
        y = df.iloc[:, 0]
        model = sm.OLS(y, X).fit()
        return model

# --- Part D: Panel Regression ---
def panel_data_regression(df):
    df = df.set_index(['ticker', 'date']).dropna()
    fe_model = PanelOLS.from_formula("ret ~ 1 + market_ret + EntityEffects", data=df).fit()
    re_model = RandomEffects.from_formula("ret ~ 1 + market_ret", data=df).fit()
    hausman_result = compare({'Fixed Effects': fe_model, 'Random Effects': re_model})
    return fe_model, re_model, hausman_result

# --- Main Program ---
if __name__ == "__main__":
    # === Part A & B ===
    final_data = pd.read_csv("Final_Data.csv", parse_dates=["Date"])
    final_data.set_index("Date", inplace=True)

    print("=== ADF Test Results ===")
    for col in final_data.columns:
        series = final_data[col].dropna()
        analyzer = TimeSeriesAnalyzer(series, col)
        analyzer.plot_series()
        adf = analyzer.adf_test()
        print(f"{col}: ADF = {adf['ADF Statistic']:.4f}, p = {adf['p-value']:.4f}, Stationary = {adf['Stationary']}")
        if not adf['Stationary']:
            diff_result = analyzer.difference_and_test()
            print(f"{col} (1st Diff): ADF = {diff_result['ADF Statistic']:.4f}, p = {diff_result['p-value']:.4f}, Stationary = {diff_result['Stationary']}")

    # === Part C ===
    panel_data = pd.read_csv("panel_data.csv", parse_dates=["date"])
    panel_data['date'] = pd.to_datetime(panel_data['date'])

    # Calculate returns
    panel_data['ret'] = panel_data.groupby('ticker')['price'].pct_change()
    panel_data['market_ret'] = panel_data['index'].pct_change()

    rr = ReturnRegressor()
    print("\n=== Return Regressions ===")
    tickers = panel_data['ticker'].unique()
    for ticker in tickers:
        if ticker == 'INDEX':
            continue
        stock_df = panel_data[panel_data['ticker'] == ticker]
        model = rr.run_regression(stock_df['ret'], stock_df['market_ret'])
        print(f"\nRegression for {ticker}:\n", model.summary())

    # === Part D ===
    print("\n=== Panel Data Regression ===")
    panel_subset = panel_data[panel_data['ticker'] != 'INDEX'][['ticker', 'date', 'ret', 'market_ret']]
    fe_model, re_model, hausman_result = panel_data_regression(panel_subset)
    print("\nFixed Effects Model:\n", fe_model.summary)
    print("\nRandom Effects Model:\n", re_model.summary)
    print("\nHausman Test:\n", hausman_result)

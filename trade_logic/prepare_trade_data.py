from binance.client import Client
import pandas as pd

client = Client()  # no API key needed for public data

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE

klines = client.get_historical_klines(
    symbol,
    interval,
    "6 days ago UTC"
)

df = pd.DataFrame(klines, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","num_trades",
    "taker_buy_base","taker_buy_quote","ignore"
])

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

df.to_csv("btcusdt_5m.csv", index=False)

print(df.head())
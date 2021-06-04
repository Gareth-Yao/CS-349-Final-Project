from train_lstm import train_lstm


period = ['1y','5y','10y']
for i in period:
    train_lstm(i, '1d')

hidden_nodes = [10, 100, 500]
for i in hidden_nodes:
    train_lstm('max', '1d', hidden_nodes = i)

tickers = ['AMZN', 'AAPL', 'BTC-USD']
for i in tickers:
    train_lstm('max', '1d', ticker = [i])

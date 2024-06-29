from datetime import datetime
from api import loader
from factor1 import MDS, AggregatedTrade
from utils import ProgressiveReplay


trade_monitor = AggregatedTrade(name='AggregatedTrade')
MDS.add_monitor(trade_monitor)

start_date = datetime(2024, 3, 8)
end_date = datetime(2024, 3, 8)
tickers = ['000004.SZ']
dtypes = ['TradeData']
replay = ProgressiveReplay(
    loader=loader,
    start_date=start_date,
    end_date=end_date,
    ticker=tickers,
    dtype=dtypes
)

for md in replay:
    MDS.on_market_data(market_data=md)

# After replay is done, generate and plot distribution
trade_monitor.plot_distribution()

# Fit and print distribution details
distribution_details = trade_monitor.fit_distribution()
print(distribution_details)


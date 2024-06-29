from datetime import datetime
from api import loader
from factor1 import MDS, AggregatedTrade
from factordesign2 import ChipDistribution
from utils import ProgressiveReplay

chip_distribution_monitor = ChipDistribution(name='ChipDistribution', sampling_interval=1, sample_size=60)
MDS.add_monitor(chip_distribution_monitor)

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
chip_distribution_monitor.plot_distribution()

# Log events to CSV
chip_distribution_monitor.log_to_csv()

# Calculate prediction power
prediction_lengths = [1, 5, 10, 30, 60]  # Example lengths in seconds
prediction_power = chip_distribution_monitor.calculate_prediction_power(prediction_lengths)
print(prediction_power)

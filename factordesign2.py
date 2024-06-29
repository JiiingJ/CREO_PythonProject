import json
import logging
import uuid
from collections import defaultdict, deque
import datetime
import plotly.graph_objects as go
import csv
from PyQuantKit import TradeData
import numpy as np
from factor1 import AggregatedTrade
from utils import ProgressiveReplay
from sampler import FixedIntervalSampler


LOGGER = logging.Logger('chip_distribution')


class ChipDistribution(AggregatedTrade, FixedIntervalSampler):
    def __init__(self, name: str, monitor_id: str = None, sampling_interval: float = 1., sample_size: int = 60):
        AggregatedTrade.__init__(self, name, monitor_id)
        FixedIntervalSampler.__init__(self, sampling_interval, sample_size)

        # Register samplers
        self.register_sampler('market_price')

        self.market_prices = defaultdict(deque)
        self.mean_prices = defaultdict(float)
        self.std_prices = defaultdict(float)
        self.upper_bounds = defaultdict(deque)
        self.lower_bounds = defaultdict(deque)
        self.timestamps = defaultdict(deque)
        self.logs = []

    def __call__(self, market_data: TradeData, **kwargs):
        super().__call__(market_data, **kwargs)
        price = market_data.price
        timestamp = market_data.timestamp
        ticker = market_data.ticker

        # Skip price 0
        if price == 0:
            return

        # Logging market price
        self.log_obs(ticker, timestamp, observation={'market_price': price})

        # Calculate mean and standard deviation
        prices = np.array(self.market_prices[ticker])
        mean_price = prices.mean()
        std_price = prices.std()

        self.mean_prices[ticker] = mean_price
        self.std_prices[ticker] = std_price

        # Calculate bounds
        upper_bound = mean_price + 2 * std_price
        lower_bound = mean_price - 2 * std_price

        self.market_prices[ticker].append(price)
        self.upper_bounds[ticker].append(upper_bound)
        self.lower_bounds[ticker].append(lower_bound)
        self.timestamps[ticker].append(timestamp)

        self.check_crossing(ticker, price, upper_bound, lower_bound, timestamp)

    def check_crossing(self, ticker, price, upper_bound, lower_bound, timestamp):
        if len(self.market_prices[ticker]) > 1:
            previous_price = self.market_prices[ticker][-2]

            if previous_price < upper_bound and price >= upper_bound:
                self.log_event(ticker, timestamp, 'crossed upper bound')
            elif previous_price > lower_bound and price <= lower_bound:
                self.log_event(ticker, timestamp, 'crossed lower bound')
            elif (previous_price >= upper_bound and price < upper_bound) or (
                    previous_price <= lower_bound and price > lower_bound):
                self.log_event(ticker, timestamp, 'returned to nominal')

    def log_event(self, ticker, timestamp, event):
        log_entry = {
            'ticker': ticker,
            'timestamp': timestamp,
            'event': event
        }
        self.logs.append(log_entry)
        LOGGER.info(f'{ticker} at {timestamp} {event}')

    def to_json(self, fmt='str') -> dict | str:
        data = {
            'market_prices': {k: list(v) for k, v in self.market_prices.items()},
            'upper_bounds': {k: list(v) for k, v in self.upper_bounds.items()},
            'lower_bounds': {k: list(v) for k, v in self.lower_bounds.items()},
            'logs': self.logs
        }
        return data if fmt == 'dict' else json.dumps(data)

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> 'ChipDistribution':
        if isinstance(json_message, (bytes, bytearray)):
            json_message = json_message.decode()
        data = json.loads(json_message) if isinstance(json_message, str) else json_message
        monitor = cls(name='ChipDistribution')
        monitor.market_prices = defaultdict(deque, {k: deque(v) for k, v in data.get('market_prices', {}).items()})
        monitor.upper_bounds = defaultdict(deque, {k: deque(v) for k, v in data.get('upper_bounds', {}).items()})
        monitor.lower_bounds = defaultdict(deque, {k: deque(v) for k, v in data.get('lower_bounds', {}).items()})
        monitor.logs = data.get('logs', [])
        return monitor

    def plot_distribution(self):
        for ticker in self.market_prices:
            prices = list(self.market_prices[ticker])
            upper_bounds = list(self.upper_bounds[ticker])
            lower_bounds = list(self.lower_bounds[ticker])
            timestamps = [i * self.sampling_interval for i in range(len(prices))]
            #timestamps = list(self.timestamps[ticker])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=prices, mode='lines', name='Market Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=timestamps, y=upper_bounds, mode='lines', name='Upper Bound', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=timestamps, y=lower_bounds, mode='lines', name='Lower Bound', line=dict(color='green', dash='dash')))

            for log in self.logs:
                if log['ticker'] == ticker:
                    index = int((log['timestamp'] - timestamps[0]) // self.sampling_interval)
                    #index = int((log['timestamp'] - 1709861400+150) // self.sampling_interval)
                    print(index, log['event'])
                    if 0 <= index < len(prices):
                        fig.add_annotation(
                            x=timestamps[index],
                            y=prices[index],
                            text=log['event'],
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )

            fig.update_layout(
                title=f'{ticker} Price and Bounds',
                xaxis_title='Time',
                yaxis_title='Price',
                barmode='overlay',
                template='plotly_white'
            )

            fig.write_html(f'{ticker}_price_bounds.html')
            fig.show()

    def log_to_csv(self, filename='chip_distribution_logs.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['ticker', 'timestamp', 'event'])
        writer.writeheader()
        writer.writerows(self.logs)

    def calculate_prediction_power(self, prediction_time_lengths):
        LOGGER.info(f'Calculating prediction power with lengths: {prediction_time_lengths}')
        prediction_power = {length: {'upper_bound_cross': 0, 'lower_bound_cross': 0, 'total_upper': 0, 'total_lower': 0}
                            for length in prediction_time_lengths}

        for log in self.logs:
            ticker = log['ticker']
            timestamp = log['timestamp']
            event = log['event']

            for length in prediction_time_lengths:
                future_timestamp = timestamp + length
                if future_timestamp in self.market_prices[ticker]:
                    future_price = self.market_prices[ticker][future_timestamp]
                    current_price = self.market_prices[ticker][timestamp]

                    if event == 'crossed upper bound':
                        prediction_power[length]['total_upper'] += 1
                        if future_price > current_price:
                            prediction_power[length]['upper_bound_cross'] += 1
                    elif event == 'crossed lower bound':
                        prediction_power[length]['total_lower'] += 1
                        if future_price < current_price:
                            prediction_power[length]['lower_bound_cross'] += 1

        for length in prediction_time_lengths:
            if prediction_power[length]['total_upper'] > 0:
                prediction_power[length]['upper_bound_cross'] /= prediction_power[length]['total_upper']
            if prediction_power[length]['total_lower'] > 0:
                prediction_power[length]['lower_bound_cross'] /= prediction_power[length]['total_lower']

        return {length: {'upper_bound_cross': prediction_power[length]['upper_bound_cross'],
                         'lower_bound_cross': prediction_power[length]['lower_bound_cross']} for length in
                prediction_time_lengths}


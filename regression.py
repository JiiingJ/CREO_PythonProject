import csv
import datetime
import os
import pathlib
import time
from collections.abc import Iterable
from typing import List, Dict

import numpy as np
import pandas as pd
from PyQuantKit import TradeData
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr

from . import LOGGER
from ..Base import GlobalStatics

ARCHIVE_DIR = pathlib.Path.home().joinpath('Documents', 'TradeDataArchive')
DATA_DIR = pathlib.Path.home().joinpath('Documents', 'TradeData')
TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE


def unzip(market_date: datetime.date, ticker: str):
    import py7zr

    archive_path = pathlib.Path(ARCHIVE_DIR, f'{market_date:%Y%m}', f'{market_date:%Y-%m-%d}.7z')
    destination_path = pathlib.Path(DATA_DIR)
    directory_to_extract = f'{market_date:%Y-%m-%d}'
    file_to_extract = f'{ticker.split(".")[0]}.csv'

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f'{archive_path} not found!')

    os.makedirs(destination_path, exist_ok=True)

    LOGGER.info(f'Unzipping {file_to_extract} from {archive_path} to {destination_path}...')

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extract(targets=[f'{directory_to_extract}/{file_to_extract}'], path=destination_path)

    return 0


def unzip_batch(market_date: datetime.date, ticker_list: Iterable[str]):
    import py7zr

    archive_path = pathlib.Path(ARCHIVE_DIR, f'{market_date:%Y%m}', f'{market_date:%Y-%m-%d}.7z')
    destination_path = pathlib.Path(DATA_DIR)
    directory_to_extract = f'{market_date:%Y-%m-%d}'

    targets = []

    for ticker in ticker_list:
        name = f'{ticker.split(".")[0]}.csv'
        destination = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

        if os.path.isfile(destination):
            continue

        targets.append(f'{directory_to_extract}/{name}')

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f'{archive_path} not found!')

    os.makedirs(destination_path, exist_ok=True)

    if not targets:
        return 0

    LOGGER.info(f'Unzipping {len(targets)} names from {archive_path} to {destination_path}...')

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extract(targets=targets, path=destination_path)

    return 0


def load_trade_data(market_date: datetime.date, ticker: str) -> List[TradeData]:
    ts = time.time()
    trade_data_list = []

    file_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

    if not os.path.isfile(file_path):
        try:
            unzip(market_date=market_date, ticker=ticker)
        except FileNotFoundError as _:
            return trade_data_list

    with open(file_path, 'r') as f:
        data_file = csv.DictReader(f)
        for row in data_file:
            trade_data = TradeData(
                ticker=ticker,
                trade_price=float(row['Price']),
                trade_volume=float(row['Volume']),
                timestamp=datetime.datetime.combine(market_date, datetime.time(*map(int, row['Time'].split(":"))), TIME_ZONE).timestamp(),
                side=row['Type'],
                buy_id=int(row['BuyOrderID']),
                sell_id=int(row['SaleOrderID'])
            )
            trade_data_list.append(trade_data)

            if DEBUG_MODE:
                if not np.isfinite(trade_data.trade_volume):
                    raise ValueError(f'Invalid trade data {trade_data}, volume = {trade_data.trade_volume}')

                if not np.isfinite(trade_data.trade_price) or trade_data.trade_price < 0:
                    raise ValueError(f'Invalid trade data {trade_data}, price = {trade_data.trade_price}')

                if trade_data.side == "":
                    raise ValueError(f'Invalid trade data {trade_data}, side = {trade_data.side}')

    LOGGER.info(f'{market_date} {ticker} trade data loaded, {len(trade_data_list):,} entries in {time.time() - ts:.3f}s.')

    return trade_data_list


def loader(market_date: datetime.date, ticker: str, dtype: str):
    if dtype == 'TradeData':
        return load_trade_data(market_date=market_date, ticker=ticker)
    else:
        raise NotImplementedError(f'API.historical does not have a loader function for {dtype}')


def calculate_trade_flow(trade_data: List[TradeData]) -> float:
    net_buy = sum(trade.trade_volume for trade in trade_data if trade.side == 'BUY')
    net_sell = sum(trade.trade_volume for trade in trade_data if trade.side == 'SELL')
    return net_buy - net_sell


def add_session_dummies(data: pd.DataFrame) -> pd.DataFrame:
    data['is_opening'] = data['Time'].apply(lambda x: x.time() < datetime.time(10, 0))
    data['is_closing'] = data['Time'].apply(lambda x: x.time() > datetime.time(14, 30))
    return data


def build_regression_model(data: pd.DataFrame, target: str):
    X = data[['trade_flow', 'is_opening', 'is_closing']]
    y = data[target]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    return model, y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, np.round(y_pred))
    ir = np.mean(y_true - y_pred) / np.std(y_true - y_pred)
    ic, _ = pearsonr(y_true, y_pred)
    ic_rank, _ = pearsonr(np.argsort(y_true), np.argsort(y_pred))

    return {
        'accuracy': acc,
        'information_ratio': ir,
        'ic': ic,
        'ic_rank': ic_rank,
    }


# Example usage
market_date = datetime.date(2023, 6, 1)
ticker = 'AAPL'

# Load trade data
trade_data = loader(market_date=market_date, ticker=ticker, dtype='TradeData')

# Convert to DataFrame for easier manipulation
data = pd.DataFrame([{
    'Time': datetime.datetime.fromtimestamp(trade.timestamp, tz=TIME_ZONE),
    'trade_volume': trade.trade_volume,
    'trade_price': trade.trade_price,
    'side': trade.side
} for trade in trade_data])

# Add trade flow and session dummies
data['trade_flow'] = calculate_trade_flow(trade_data)
data = add_session_dummies(data)

# Define prediction targets (for example, next period's price change)
data['price_pct_chg'] = data['trade_price'].pct_change().shift(-1)

# Drop rows with NaN values
data = data.dropna()

# Build and evaluate the model for different targets
model, y_pred = build_regression_model(data, 'price_pct_chg')

# Calculate metrics
metrics = calculate_metrics(data['price_pct_chg'].values, y_pred)

print(f'Metrics: {metrics}')

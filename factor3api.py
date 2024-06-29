import csv
import datetime
import os
import pathlib
import time
from collections.abc import Iterable
from typing import List, Dict

import numpy as np
from PyQuantKit import TradeData

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


def calculate_trade_imbalance(trade_flow: float, total_volume: float) -> float:
    return trade_flow / total_volume if total_volume != 0 else 0


def calculate_boosted_trade_imba(trade_imbalance: float, price_pct_chg: float) -> float:
    uu = trade_imbalance if trade_imbalance > 0 and price_pct_chg > 0 else 0
    dd = trade_imbalance if trade_imbalance < 0 and price_pct_chg < 0 else 0
    return uu - dd


def calculate_trade_imba_slope(trade_imbalance_list: List[float]) -> float:
    x = np.arange(len(trade_imbalance_list))
    y = np.array(trade_imbalance_list)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def calculate_entropy(trade_imba_matrix: np.ndarray) -> float:
    joint_dist = trade_imba_matrix / np.sum(trade_imba_matrix)
    entropy = -np.nansum(joint_dist * np.log2(joint_dist))
    return entropy


def calculate_ic(factor_values: List[float], returns: List[float]) -> float:
    return np.corrcoef(factor_values, returns)[0, 1]


def calculate_ic_rank(factor_values: List[float], returns: List[float]) -> float:
    return np.corrcoef(np.argsort(factor_values), np.argsort(returns))[0, 1]


def calculate_auc_roc(ic_quantile: List[float], ic_rank_quantile: List[float], acc_quantile: List[float]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    return {
        'auc_ic_quantile': roc_auc_score(ic_quantile, np.arange(len(ic_quantile))),
        'auc_ic_rank_quantile': roc_auc_score(ic_rank_quantile, np.arange(len(ic_rank_quantile))),
        'auc_acc_quantile': roc_auc_score(acc_quantile, np.arange(len(acc_quantile))),
    }


# Example usage:
market_date = datetime.date(2023, 6, 1)
ticker = 'AAPL'

# Load trade data
trade_data = loader(market_date=market_date, ticker=ticker, dtype='TradeData')

# Calculate trade flow
trade_flow = calculate_trade_flow(trade_data)
total_volume = sum(trade.trade_volume for trade in trade_data)
trade_imbalance = calculate_trade_imbalance(trade_flow, total_volume)
price_pct_chg = 0.02  # Example price percentage change
boosted_trade_imba = calculate_boosted_trade_imba(trade_imbalance, price_pct_chg)

# Example trade imbalance list for slope calculation
trade_imbalance_list = [trade_imbalance] * 10  # Replace with actual list
trade_imba_slope = calculate_trade_imba_slope(trade_imbalance_list)

# Example trade imbalance matrix for entropy calculation
trade_imba_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])  # Replace with actual matrix
entropy = calculate_entropy(trade_imba_matrix)

# Example factor values and returns for IC and IC Rank calculations
factor_values = [0.1, 0.2, 0.3, 0.4, 0.5]
returns = [0.05, 0.1, 0.15, 0.2, 0.25]
ic = calculate_ic(factor_values, returns)
ic_rank = calculate_ic_rank(factor_values, returns)

# Example quantiles for AUC ROC calculation
ic_quantile = [0.1, 0.2, 0.3, 0.4, 0.5]
ic_rank_quantile = [0.1, 0.2, 0.3, 0.4, 0.5]
acc_quantile = [0.1, 0.2, 0.3, 0.4, 0.5]
auc_roc_scores = calculate_auc_roc(ic_quantile, ic_rank_quantile, acc_quantile)

print(f'Trade Flow: {trade_flow}')
print(f'Trade Imbalance: {trade_imbalance}')
print(f'Boosted Trade Imba: {boosted_trade_imba}')
print(f'Trade Imba Slope: {trade_imba_slope}')
print(f'Entropy: {entropy}')
print(f'IC: {ic}')
print(f'IC Rank: {ic_rank}')
print(f'AUC ROC Scores: {auc_roc_scores}')

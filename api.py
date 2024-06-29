import os
import csv
import datetime
# from datetime import datetime
# from datetime import date
from PyQuantKit import TradeData, TickData, OrderBook


def loader(market_date: datetime.date, ticker: str, dtype: str) -> list:
    base_dir = f"/Users/wangyujing/Desktop/面试/Res/TransactionData/{market_date.strftime('%Y-%m-%d')}"

    if dtype == 'TradeData':
        file_path = os.path.join(base_dir, 'transactions', f'{ticker}.csv')
        return load_trade_data(file_path)

    elif dtype == 'TickData':
        file_path = os.path.join(base_dir, 'ticks', f'{ticker}.csv')
        # print( file_path)
        return load_tick_data(file_path)

    elif dtype == 'TransactionData':
        transaction_path = os.path.join(base_dir, 'transactions', f'{ticker}.csv')
        orderbook_path = os.path.join(base_dir, 'orders', f'{ticker}.csv')
        return load_transaction_data(transaction_path, orderbook_path)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def load_trade_data(file_path: str) -> list[TradeData]:
    trade_data_list = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp_str = row['datetime']
            dt = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
            unix_timestamp = dt.timestamp()
            # print(dt1.millisecond)
            side_map = {0: 0, 1: 1, 2: -1}

            trade_data = TradeData(
                ticker=row['symbol'],
                timestamp=unix_timestamp,
                price=float(row['price']),
                volume=int(row['volume']),
                side=side_map[int(row['direction'])],
                #datetime = unix_timestamp
                # buy_id = int(row['bid_order_id']),
                # sell_id=int(row['ask_order_id']),
                # transaction_id= int(row['record_id'])
                # notional =
                # multiplier =

            )
            trade_data_list.append(trade_data)
    return trade_data_list


def load_tick_data(file_path: str) -> list[TickData]:
    tick_data_list = []
    # bid = []
    # ask = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp_str = row['datetime']
            dt = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
            unix_timestamp = dt.timestamp()
            side_map = {0: 0, 1: 1, 2: -1}

            order_book = OrderBook(
                ticker=row['symbol'],
                timestamp=unix_timestamp,
                # price=float(row['price']),
                # volume=int(row['volume']),
                # side=side_map[int(row['direction'])],
                # buy_id=row['record_id']
                bid=int(row['bid_volume_1']),
                ask=int(row['ask_volume_1'])
                # bid=int(row['bid_volume_1']),
                # ask_price=float(row['ask_price_1']),
                # bid_price=float(row['bid_price_1']),
                # ask_volume=int(row['ask_volume_1']),
                # bid_volume=int(row['bid_volume_1']),
            )

            tick_data = TickData(
                ticker=row['symbol'],
                timestamp=unix_timestamp,
                last_price=float(row['close']),
                volume=int(row['volume']),
                order_book=order_book
            )

            tick_data_list.append(tick_data)
    return tick_data_list


def load_transaction_data(transaction_path: str, orderbook_path: str) -> list:
    transaction_data_list = load_trade_data(transaction_path)
    orderbook_data_list = []
    print(orderbook_path)
    with open(orderbook_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp_str = row['datetime']
            dt = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S%f')
            unix_timestamp = dt.timestamp()
            # print(dt1.millisecond)
            side_map = {0: 0, 1: 1, 2: -1}

            # order_book = OrderBook(
            # ticker=row['symbol'],
            # timestamp=row['datetime'],
            # price=float(row['price']),
            # volume=int(row['volume']),
            # side=side_map[int(row['direction'])],
            # buy_id = int(row['bid_order_id']),
            # sell_id=int(row['ask_order_id']),
            # transaction_id= int(row['record_id'])
            # notional =
            # multiplier =
            # ask_price=float(row['ask_price']),
            # bid_price=float(row['bid_price']),
            # ask_volume=int(row['ask_volume']),
            # bid_volume=int(row['bid_volume']),
            # Add other necessary fields here
            # )
            # orderbook_data_list.append(order_book)

    # Assuming that transaction data can use orderbook in some way
    return transaction_data_list


def test_loader():
    market_date = datetime.date(2024, 3, 8)
    ticker = '000004.SZ'

    trade_data = loader(market_date, ticker, 'TradeData')
    print(f'Loaded {len(trade_data)} trade data records')

    tick_data = loader(market_date, ticker, 'TickData')
    print(f'Loaded {len(tick_data)} tick data records')

    transaction_data = loader(market_date, ticker, 'TransactionData')
    print(f'Loaded {len(transaction_data)} transaction data records')


test_loader()

market_date = datetime.date(2024, 3, 8)
# market_date=date.fromisoformat('20240308')
ticker = '000004.sz'
data = loader(market_date, ticker, 'TransactionData')
print(data[1060])
# print(data[1060].price)
# print(data[1060].side)
# print(data[1060].ticker)
# print(datetime.datetime.fromtimestamp(data[1060].timestamp).year)
# print(datetime.datetime.fromtimestamp(data[1060].timestamp).microsecond//1000)
# print(dir(data[60]))

print(dir(TradeData))
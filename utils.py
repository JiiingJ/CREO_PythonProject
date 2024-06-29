import logging
import abc
import datetime
#from datetime  import datetime
import inspect
from typing import Iterable, Callable, List, Union
from PyQuantKit import TickData, TradeData, OrderBook, Progress
from api import loader
import sys

LOGGER = logging.Logger('replay')


class Replay(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __next__(self): ...

    @abc.abstractmethod
    def __iter__(self): ...


class ProgressiveReplay(Replay):
    """
    progressively loading and replaying market data

    requires arguments
    loader: a data loading function. Expect loader = Callable(market_date: datetime.date, ticker: str, dtype: str| type) -> dict[any, MarketData]
    start_date & end_date: the given replay period
    or calendar: the given replay calendar.

    accepts kwargs:
    ticker / tickers: the given symbols to replay, expect a str| list[str]
    dtype / dtypes: the given dtype(s) of symbol to replay, expect a str | type, list[str | type]. default = all, which is (TradeData, TickData, OrderBook)
    subscription / subscribe: the given ticker-dtype pair to replay, expect a list[dict[str, str | type]]
    """

    def __init__(
            self,
            loader: Callable[[datetime.date, str, Union[str, type]], List[Union[TradeData, TickData, OrderBook]]],
            **kwargs
    ):
        self.loader = loader
        self.start_date: datetime.date | None = kwargs.pop('start_date', None)
        self.end_date: datetime.date | None = kwargs.pop('end_date', None)
        self.calendar: list[datetime.date] | None = kwargs.pop('calendar', None)

        self.eod = kwargs.pop('eod', None)
        self.bod = kwargs.pop('bod', None)

        self.replay_subscription = {}
        self.replay_calendar = []
        self.replay_task = []

        self.date_progress = 0
        self.task_progress = 0
        self.progress = Progress(tasks=1, **kwargs)

        tickers: list[str] = kwargs.pop('ticker', kwargs.pop('tickers', []))
        dtypes: list[Union[str, type]] = kwargs.pop('dtype', kwargs.pop('dtypes', [TradeData, OrderBook, TickData]))

        if not all(
                [arg_name in inspect.getfullargspec(loader).args for arg_name in ['market_date', 'ticker', 'dtype']]):
            raise TypeError('loader function has 3 requires args, market_date, ticker and dtype.')

        if isinstance(tickers, str):
            tickers = [tickers]
        elif isinstance(tickers, Iterable):
            tickers = list(tickers)
        else:
            raise TypeError(f'Invalid ticker {tickers}, expect str or list[str]')

        if isinstance(dtypes, str) or inspect.isclass(dtypes):
            dtypes = [dtypes]
        elif isinstance(dtypes, Iterable):
            dtypes = list(dtypes)
        else:
            raise TypeError(f'Invalid dtype {dtypes}, expect str or list[str]')

        for ticker in tickers:
            for dtype in dtypes:
                self.add_subscription(ticker=ticker, dtype=dtype)

        subscription = kwargs.pop('subscription', kwargs.pop('subscribe', []))

        if isinstance(subscription, dict):
            subscription = [subscription]

        for sub in subscription:
            self.add_subscription(**sub)

        self.reset()

    def add_subscription(self, ticker: str, dtype: Union[type, str]):
        if isinstance(dtype, str):
            pass
        elif inspect.isclass(dtype):
            dtype = dtype.__name__
        else:
            raise ValueError(f'Invalid dtype {dtype}, expect str or class.')

        topic = f'{ticker}.{dtype}'
        self.replay_subscription[topic] = (ticker, dtype)

    def remove_subscription(self, ticker: str, dtype: Union[type, str]):
        if isinstance(dtype, str):
            pass
        else:
            dtype = dtype.__name__

        topic = f'{ticker}.{dtype}'
        self.replay_subscription.pop(topic, None)

    def reset(self):
        if self.calendar is None:
            md = self.start_date
            self.replay_calendar.clear()

            while md <= self.end_date:
                self.replay_calendar.append(md)
                md += datetime.timedelta(days=1)

        elif callable(self.calendar):
            self.replay_calendar = self.calendar(start_date=self.start_date, end_date=self.end_date)
        else:
            self.replay_calendar = self.calendar

        self.date_progress = 0
        self.progress.reset()

    def next_trade_day(self):
        if self.date_progress < len(self.replay_calendar):
            market_date = self.replay_calendar[self.date_progress]
            self.progress.prompt = f'Replay {market_date:%Y-%m-%d} ({self.date_progress + 1} / {len(self.replay_calendar)}):'
            if self.bod:
                self.bod(market_date=market_date, replay=self)
            for topic in self.replay_subscription:
                ticker, dtype = self.replay_subscription[topic]
                LOGGER.info(f'{self} loading {market_date} {ticker} {dtype}')
                data = self.loader(market_date=market_date, ticker=ticker, dtype=dtype)

                if isinstance(data, dict):
                    self.replay_task.extend(list(data.values()))
                elif isinstance(data, (list, tuple)):
                    self.replay_task.extend(data)

            self.date_progress += 1
            if self.eod:
                self.eod(market_date=market_date, replay=self)
        else:
            raise StopIteration()

        self.replay_task.sort(key=lambda x: x.timestamp)

    def next_task(self):
        if self.task_progress < len(self.replay_task):
            task = self.replay_task[self.task_progress]
            self.task_progress += 1
            return task
        else:
            self.replay_task.clear()
            self.task_progress = 0
            self.next_trade_day()
            return self.next_task()

    def __next__(self):
        try:
            return self.next_task()
        except StopIteration:
            if not self.progress.is_done:
                self.progress.done_tasks = 1
                self.progress.output()

            self.reset()
            raise StopIteration()

    def __iter__(self):
        self.reset()
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}{{id={id(self)}, from={self.start_date}, to={self.end_date}}}'



def test_progressive_replay():

    def bod(market_date, replay):
        print(f"BOD for {market_date}")

    def eod(market_date, replay):
        print(f"EOD for {market_date}")

    start_date = datetime.date(2024, 3, 8)
    end_date = datetime.date(2024, 3, 8)
    #start_date = datetime.date.isoformat(args[0])
    #end_date = datetime.date.isoformat(args[1])
    tickers = ['000004.SZ']
    dtypes = ['TradeData']

    replay = ProgressiveReplay(
        loader=loader,
        start_date=start_date,
        end_date=end_date,
        ticker=tickers,
        dtype=dtypes,
        bod=bod,
        eod=eod
    )

    for task in replay:
        print("Replaying task:", task)


if __name__ == "__main__":
    #test_progressive_replay(sys.argv[1:])
    test_progressive_replay()







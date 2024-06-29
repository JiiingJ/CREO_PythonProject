import abc
import enum
import json
import logging
from collections import deque
from typing import Self

LOGGER = logging.Logger('sampler')


class SamplerMode(enum.Enum):
  update = 'update'
  accumulate = 'accumulate'


class FixedIntervalSampler(object, metaclass=abc.ABCMeta):
  """
  Abstract base class for a fixed interval sampler designed for financial usage.

  Args:
  - sampling_interval (float): Time interval between consecutive samples. Default is 1.
  - sample_size (int): Number of samples to be stored. Default is 60.

  Attributes:
  - sampling_interval (float): Time interval between consecutive samples.
  - sample_size (int): Number of samples to be stored.

  Warnings:
  - If `sampling_interval` is not positive, a warning is issued.
  - If `sample_size` is less than 2, a warning is issued according to Shannon's Theorem.

  Methods:
  - log_obs(ticker: str, value: float, timestamp: float, storage: Dict[str, Dict[float, float]])
      Logs an observation for the given ticker at the specified timestamp.

  - on_entry_added(ticker: str, key, value)
      Callback method triggered when a new entry is added.

  - on_entry_updated(ticker: str, key, value)
      Callback method triggered when an existing entry is updated.

  - on_entry_removed(ticker: str, key, value)
      Callback method triggered when an entry is removed.

  - clear()
      Clears all stored data.

  Notes:
  - Subclasses must implement the abstract methods: on_entry_added, on_entry_updated, on_entry_removed.

  """

  def __init__(self, sampling_interval: float = 1., sample_size: int = 60):
    """
    Initialize the FixedIntervalSampler.

    Parameters:
    - sampling_interval (float): Time interval between consecutive samples (in seconds). Default is 1.
    - sample_size (int): Number of samples to be stored. Default is 60.
    """
    self.sampling_interval = sampling_interval
    self.sample_size = sample_size

    self.sample_storage = getattr(self, 'sample_storage', {})  # to avoid de-reference the dict using nested inheritance

    # Warning for sampling_interval
    if sampling_interval <= 0:
      LOGGER.warning(f'{self.__class__.__name__} should have a positive sampling_interval')

    # Warning for sample_interval by Shannon's Theorem
    if sample_size <= 2:
      LOGGER.warning(f"{self.__class__.__name__} should have a larger sample_size, by Shannon's Theorem, sample_size should be greater than 2")

  def register_sampler(self, name: str, mode: str | SamplerMode = 'update') -> dict:
    if name in self.sample_storage:
      LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
      return self.sample_storage[name]

    if isinstance(mode, SamplerMode):
      mode = mode.value

    if mode not in ['update', 'accumulate']:
      raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

    sample_storage = self.sample_storage[name] = dict(
      storage={},
      index={},
      mode=mode
    )

    return sample_storage

  def get_sampler(self, name: str) -> dict[str, deque]:
    if name not in self.sample_storage:
      raise ValueError(f'name {name} not found in {self.__class__.__name__}!')

    return self.sample_storage[name]['storage']

  def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
    observation_copy = {}

    if observation is not None:
      observation_copy.update(observation)

    observation_copy.update(kwargs)

    idx = timestamp // self.sampling_interval

    for obs_name, obs_value in observation_copy.items():
      if obs_name not in self.sample_storage:
        raise ValueError(f'Invalid observation name {obs_name}')

      sampler = self.sample_storage[obs_name]
      storage: dict[str, deque] = sampler['storage']
      indices: dict = sampler['index']
      mode = sampler['mode']

      if ticker in storage:
        obs_storage = storage[ticker]
      elif auto_register:
        obs_storage = storage[ticker] = deque(maxlen=self.sample_size)
      else:
        LOGGER.warning(f'Ticker {ticker} not registered in sampler {obs_name}, perhaps the subscription has changed?')
        continue

      last_idx = indices.get(ticker, 0)

      if idx > last_idx:
        obs_storage.append(obs_value)
        indices[ticker] = idx
        self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
      else:
        if mode == 'update':
          last_obs = obs_storage[-1] = obs_value
        elif mode == 'accumulate':
          last_obs = obs_storage[-1] = obs_value + obs_storage[-1]
        else:
          raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

        self.on_entry_updated(ticker=ticker, name=obs_name, value=last_obs)

  def on_entry_added(self, ticker: str, name: str, value):
    """
    Callback method triggered when a new entry is added.

    Parameters:
    - ticker (str): Ticker symbol for the added entry.
    - key: Key for the added entry.
    - value: Value of the added entry.
    """
    pass

  def on_entry_updated(self, ticker: str, name: str, value):
    """
    Callback method triggered when an existing entry is updated.

    Parameters:
    - ticker (str): Ticker symbol for the updated entry.
    - key: Key for the updated entry.
    - value: Updated value of the entry.
    """
    pass

  def to_json(self, fmt='str', **kwargs) -> str | dict:
    data_dict = dict(
      sampling_interval=self.sampling_interval,
      sample_size=self.sample_size,
      sample_storage={name: dict(storage={ticker: list(dq) for ticker, dq in value['storage'].items()},
                                 index=value['index'],
                                 mode=value['mode'])
                      for name, value in self.sample_storage.items()}
    )

    if fmt == 'dict':
      return data_dict
    elif fmt == 'str':
      return json.dumps(data_dict, **kwargs)
    else:
      raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

  @classmethod
  def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
    if isinstance(json_message, dict):
      json_dict = json_message
    else:
      json_dict = json.loads(json_message)

    self = cls(
      sampling_interval=json_dict['sampling_interval'],
      sample_size=json_dict['sample_size']
    )

    for name, sampler in json_dict['sample_storage'].items():
      mode = sampler['mode']
      new_sampler = self.register_sampler(name=name, mode=mode)
      new_sampler['index'].update(sampler['index'])

      for ticker, data in sampler['storage'].items():
        if ticker in new_sampler:
          new_sampler['storage'][ticker].extend(data)
        else:
          new_sampler['storage'][ticker] = deque(data, maxlen=self.sample_size)

    return self

  def clear(self):
    """
    Clears all stored data.
    """
    for name, sample_storage in self.sample_storage.items():
      for ticker, dq in sample_storage['storage'].items():
        dq.clear()

      self.sample_storage[name]['index'].clear()

    # using this code will require the sampler to be registered again.
    self.sample_storage.clear()

  def loc_obs(self, name: str, ticker: str, index: int | slice = None) -> float | list[float]:
    sampler = self.get_sampler(name=name)
    observation = sampler.get(ticker, [])

    if index is None:
      return list(observation)
    else:
      return list(observation)[index]

  def active_obs(self, name: str) -> dict[str, float]:
    sampler = self.get_sampler(name=name)
    last_obs = {}

    for ticker, observation in sampler.items():
      if observation:
        last_obs[ticker] = observation[-1]

    return last_obs
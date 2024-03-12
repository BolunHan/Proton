from __future__ import annotations

import abc
import enum
import json
import pickle
import time
import traceback
from collections import deque
from ctypes import c_wchar, c_bool, c_double
from multiprocessing import RawArray, RawValue, Semaphore, Process, shared_memory, Lock
from typing import Self, Callable

import numpy as np
import pandas as pd
from PyQuantKit import MarketData, TickData, TradeData, TransactionData, OrderBook, BarData, OrderBookBuffer, BarDataBuffer, TickDataBuffer, TransactionDataBuffer
from Quark.Calibration.dummies import is_market_session
from Quark.Factor.memory_core import SyncMemoryCore, NamedVector
from Quark.Factor.utils import FactorMonitor as FactorMonitorBase, ConcurrentMonitorManager, IndexWeight as IndexWeightBase, Synthetic, EMA, SamplerMode, FixedIntervalSampler, FixedVolumeIntervalSampler, AdaptiveVolumeIntervalSampler
from scipy.stats import rankdata
from . import collect_factor
from .. import LOGGER

ALPHA_05 = 0.9885  # alpha = 0.5 for each minute
ALPHA_02 = 0.9735  # alpha = 0.2 for each minute
ALPHA_01 = 0.9624  # alpha = 0.1 for each minute
ALPHA_001 = 0.9261  # alpha = 0.01 for each minute
ALPHA_0001 = 0.8913  # alpha = 0.001 for each minute


class IndexWeight(IndexWeightBase):
    def information_coefficient(self, factor_value: dict[str, float], pred_value: dict[str, float], mode: str = 'spearman') -> float:
        """ref https://notebook.community/quantopian/research_public/notebooks/lectures/Factor_Analysis/notebook"""
        selected_ticker = set(self) | set(factor_value) | set(pred_value)

        if len(selected_ticker) < 3:
            return np.nan

        selected_factor = np.array([factor_value[ticker] for ticker in selected_ticker])
        selected_pred = np.array([pred_value[ticker] for ticker in selected_ticker])

        if mode == 'pearson':
            corr = np.corrcoef(selected_factor, selected_pred)
            ic = corr[0, 1]
        elif mode == 'spearman':
            corr = np.corrcoef(rankdata(selected_factor), rankdata(selected_pred))
            ic = corr[0, 1]
        else:
            raise NotImplementedError(f'Invalid mode {mode}, expect "spearman" or "pearson".')

        return ic

    def ic(self, factor_value: dict[str, float], pred_value: dict[str, float]) -> float:
        """
        the Pearson correlation coefficient

        we do not use the covariance definition as that will make horizontal comparison not feasible.
        """
        return self.information_coefficient(factor_value=factor_value, pred_value=pred_value, mode='pearson')

    def ic_rank(self, factor_value: dict[str, float], pred_value: dict[str, float]) -> float:
        return self.information_coefficient(factor_value=factor_value, pred_value=pred_value, mode='spearman')


class FactorMonitor(FactorMonitorBase, metaclass=abc.ABCMeta):
    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)

        if isinstance(self, FixedIntervalCompressor):
            data_dict.update(
                FixedIntervalCompressor.to_json(self=self, fmt='dict')
            )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def update_from_json(self, json_dict: dict) -> Self:
        super().update_from_json(json_dict=json_dict)

        if isinstance(self, FixedIntervalCompressor):
            self: FixedIntervalCompressor

            for name, data in json_dict['compressor'].items():
                new_compressor = self.register_compressor(name=name)
                new_compressor.extend(data)

        return self

    def clear(self) -> None:
        super().clear()

        if isinstance(self, FixedIntervalCompressor):
            FixedIntervalCompressor.clear(self=self)


class FixedIntervalCompressor(object, metaclass=abc.ABCMeta):
    class Compressor(deque):
        def __init__(self, name: str, size: int, *args, **kwargs):
            self.name = name
            self.index: int = 0

            super().__init__(*args, **kwargs, maxlen=size)

        def clear(self):
            super().clear()
            self.index = 0

        def slope(self) -> float:
            if len(self) < 3:
                return np.nan

            x = list(range(len(self)))
            x = np.vstack([x, np.ones(len(x))]).T
            y = np.array(self)

            slope, c = np.linalg.lstsq(x, y, rcond=None)[0]

            return slope

        def ema(self, window: int = None, alpha: float = None) -> float:
            if not self:
                return np.nan

            ema = self[0]

            for current in self[0:]:
                ema = EMA.calculate_ema(value=current, memory=ema, alpha=alpha, window=window)

            return ema

        def diff(self, order: int = 1) -> float:
            if len(self) < order + 2:
                return np.nan

            diff = np.array(self)

            for _ in range(order):
                diff = np.diff(diff)

            return np.mean(diff)

        def diff_ema(self, order: int, alpha: float = None, window: int = None) -> float:
            if len(self) < order + 2:
                return np.nan

            diff = np.array(self)

            for _ in range(order):
                diff = np.diff(diff)

            ema = diff[0]

            for current in diff[0:]:
                ema = EMA.calculate_ema(value=current, memory=ema, alpha=alpha, window=window)

            return ema

        def ma(self, window: int) -> float:
            if not self:
                return np.nan

            return np.mean(list(self)[-window:])

        def macd(self, short_window=12, long_window=26, signal_window=9) -> float:
            from Quark.Factor.ta import MACD

            macd = MACD(short_window=short_window, long_window=long_window, signal_window=signal_window)

            for value in self:
                macd.calculate_macd(price=value)

            return macd.macd_diff

    def __init__(self, compress_interval: float, compress_size: int):
        self.compress_interval = compress_interval
        self.compress_size = compress_size
        self.compressor: dict[str, FixedIntervalCompressor.Compressor] = {}

    def register_compressor(self, name: str):
        if name in self.compressor:
            LOGGER.error(f'Name {name} already registered!')
            return self.compressor[name]

        _ = self.compressor[name] = self.Compressor(name=name, size=self.compress_size)
        return _

    def update_compressor(self, timestamp: float, observation: dict[str, float] | float = None, auto_register: bool = True, **kwargs):
        observation_copy = {}

        if observation is None:
            pass
        elif isinstance(observation, (float, int, bool)):
            if isinstance(self, FactorMonitor):
                name = self.name.removeprefix('Monitor.')
            else:
                raise ValueError(f'expect {self} a FactorMonitor, got {type(self)}')

            observation_copy[name] = observation
        elif isinstance(observation, dict):
            observation_copy.update(observation)
        else:
            raise TypeError(f'Expect observation a Dict[str, float], got {type(observation)}')

        observation_copy.update(kwargs)

        idx = timestamp // self.compress_interval

        for obs_name, obs_value in observation_copy.items():
            if obs_name not in self.compressor:
                if auto_register:
                    self.register_compressor(name=obs_name)
                else:
                    raise ValueError(f'Invalid compressor name {obs_name}')

            compressor = self.compressor[obs_name]
            last_idx = compressor.index

            if idx > last_idx:
                compressor.append(obs_value)
                compressor.index = idx
            elif compressor:
                compressor[-1] = obs_value

    def auto_compress(self, timestamp: float, generator: Callable[[], float | dict[str, float]], override: bool = False):
        if not isinstance(self, FactorMonitor):
            LOGGER.warning(f'Caution! The auto_compress method is designed for FactorMonitor, got {type(self)} instead!')

        indices = {name: compressor.index for name, compressor in self.compressor.items()}
        min_index = min(indices.values()) if indices else 0
        max_index = max(indices.values()) if indices else 0
        idx = int(timestamp // self.compress_interval)

        # fill the missing observation with the latest one
        for name, index in indices.items():
            compressor = self.compressor[name]

            if not compressor:
                continue

            while index < idx - 1:
                compressor.append(compressor[-1])
                compressor.index += 1

        # log new observations
        if override or idx > max_index:
            observation = generator()
            self.update_compressor(timestamp=timestamp, observation=observation, auto_register=True)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            compress_interval=self.compress_interval,
            compress_size=self.compress_size,
            compressor={name: list(compressor) for name, compressor in self.compressor.items()}
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
            compress_interval=json_dict['compress_interval'],
            compress_size=json_dict['compress_size']
        )

        for name, data in json_dict['compressor'].items():
            new_compressor = self.register_compressor(name=name)
            new_compressor.extend(data)

        return self

    def clear(self):
        for compressor in self.compressor.values():
            compressor.clear()

        self.compressor.clear()


__all__ = ['FactorMonitor', 'ConcurrentMonitorManager',
           'EMA', 'ALPHA_05', 'ALPHA_02', 'ALPHA_01', 'ALPHA_001', 'ALPHA_0001',
           'Synthetic', 'IndexWeight',
           'SamplerMode', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler',
           'FixedIntervalCompressor']

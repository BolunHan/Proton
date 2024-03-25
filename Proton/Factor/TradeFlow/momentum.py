"""
Proxy factor for support and reversion.

ref: [《量化择时系列（1）：金融工程视角下的技术择时艺术》](https://www.research.cicc.com/data/tSearch?query=%E9%87%8F%E5%8C%96%E6%8B%A9%E6%97%B6%E7%B3%BB%E5%88%97%20%E6%8B%A9%E6%97%B6%E8%89%BA%E6%9C%AF&type=REPORT_CHART)

QRS: de

"""

import json
from collections import deque
from typing import Iterable

import numpy as np
from PyQuantKit import TradeData, TransactionData

from .. import Synthetic, FactorMonitor, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, LOGGER


class MomentumSupportMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, name: str = 'Monitor.QRS', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price', mode='update')
        self.register_sampler(name='high_price', mode='max')
        self.register_sampler(name='low_price', mode='min')
        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='trade_flow', mode='accumulate')

        self._qrs_analysis: dict[str, dict[str, deque]] = dict(
            r2={},
            beta={}
        )

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        ticker = trade_data.ticker
        timestamp = trade_data.timestamp
        market_price = trade_data.market_price
        volume = trade_data.volume
        side = trade_data.side.sign

        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, high_price=market_price, low_price=market_price, volume=volume, trade_flow=volume * side)

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'trade_flow':  # the trade_flow is log at last, which is why used as a filter.
            return

        qrs_analysis = self.qrs(ticker=ticker, drop_last=True)

        if ticker not in qrs_analysis:
            return

        slope, r_squared = qrs_analysis[ticker]

        if not (np.isfinite(slope) and np.isfinite(r_squared)):
            return

        if ticker in self._qrs_analysis['r2']:
            qrs_r2 = self._qrs_analysis['r2'][ticker]
            qrs_beta = self._qrs_analysis['beta'][ticker]
        elif not self.use_shm:
            qrs_r2 = self._qrs_analysis['r2'][ticker] = deque(maxlen=max(5, int(self.sample_size / 2)))
            qrs_beta = self._qrs_analysis['beta'][ticker] = deque(maxlen=max(5, int(self.sample_size / 2)))
        else:
            LOGGER.info(f'Ticker {ticker} not registered in {self.name} for _historical_trade_imbalance, perhaps the subscription has been changed?')
            return

        qrs_r2.append(r_squared)
        qrs_beta.append(slope)

    def qrs(self, ticker: str = None, drop_last: bool = True) -> dict[str, tuple[float, float]]:
        low_dict = self.get_sampler(name='low_price')
        high_dict = self.get_sampler(name='high_price')
        price_dict = self.get_sampler(name='price')
        volume_dict = self.get_sampler(name='volume')
        trade_flow_dict = self.get_sampler(name='trade_flow')
        result = {}

        if ticker is None:
            tasks = list(volume_dict)
        elif isinstance(ticker, str):
            tasks = [ticker]
        elif isinstance(ticker, Iterable):
            tasks = list(ticker)
        else:
            raise TypeError(f'Invalid ticker {ticker}, expect str, list[str] or None.')

        for ticker in tasks:
            price_vector = list(price_dict[ticker])[:-1]
            low_vector = list(high_dict[ticker])[1:]
            high_vector = list(low_dict[ticker])[1:]
            volume_vector = list(volume_dict[ticker])[1:]
            previous_volume_vector = list(volume_dict[ticker])[:-1]
            trade_flow_vector = list(trade_flow_dict[ticker])[1:]

            if not price_vector:
                continue

            if drop_last:
                price_vector.pop(-1)
                low_vector.pop(-1)
                high_vector.pop(-1)
                volume_vector.pop(-1)
                previous_volume_vector.pop(-1)
                trade_flow_vector.pop(-1)

            # adjusted price, this changes the regression parameter slightly.
            low_vector = np.array(low_vector)
            # low_vector = np.array(low_vector) / np.array(price_vector)
            high_vector = np.array(high_vector)
            # high_vector = np.array(high_vector) / np.array(price_vector)
            flow_vector = np.array(trade_flow_vector)
            # flow_vector = np.array(trade_flow_vector) / np.array(volume_vector)
            volume_vector = np.array(volume_vector)
            # volume_vector = np.array(volume_vector) / np.array(previous_volume_vector)

            # Add a column of ones to x to represent the intercept term
            x = low_vector.reshape(-1, 1)
            x_ext = np.column_stack((x, flow_vector, volume_vector))
            y = np.array(high_vector)
            mask = np.all(np.isfinite(x), axis=1)

            x = x[mask]
            x_ext = x_ext[mask]
            y = y[mask]

            if np.sum(mask) < 4:
                continue

            q, r, *_ = np.linalg.qr(x_ext)
            q *= r[0, 0]

            x = np.column_stack((q, np.ones_like(x)))
            # x = np.column_stack((x, np.ones_like(x)))

            # Calculate the coefficients (including intercept) using linear least squares
            coefficients, residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)

            # Extract the intercept and slope from the coefficients
            beta, *_, intercept = coefficients
            # beta /= r[0, 0]  # restore the scale of x matrix

            # Calculate R^2
            if not (total_sum_squares := np.sum((y - np.mean(y)) ** 2)):
                r_squared = np.nan
            else:
                residual_sum_squares = np.sum(residuals)
                r_squared = 1 - (residual_sum_squares / total_sum_squares)

            result[ticker] = (beta, r_squared)

        return result

    def slope(self, key: str) -> dict[str, float]:
        slope_dict = {}
        for ticker, qrs_analysis in self._qrs_analysis[key].items():
            if len(qrs_analysis) < 3:
                slope = np.nan
            else:
                x = list(range(len(qrs_analysis)))
                x = np.vstack([x, np.ones(len(x))]).T
                y = np.array(qrs_analysis)

                slope, c = np.linalg.lstsq(x, y, rcond=None)[0]

            slope_dict[ticker] = slope

        return slope_dict

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            qrs_analysis={name: {ticker: list(dq) for ticker, dq in storage.items()} for name, storage in self._qrs_analysis.items()}
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def update_from_json(self, json_dict: dict):
        super().update_from_json(json_dict=json_dict)

        for name, storage in json_dict['qrs_analysis'].items():
            if name not in self._qrs_analysis:
                self._qrs_analysis[name] = {}

            for ticker, data in storage.items():
                if ticker in self._qrs_analysis[name]:
                    self._qrs_analysis[name][ticker].extend(data)
                else:
                    self._qrs_analysis[name][ticker] = deque(data, maxlen=max(5, int(self.sample_size / 2)))

        return self

    def clear(self):
        super().clear()

        for storage in self._qrs_analysis.values():
            storage.clear()

        self.register_sampler(name='price', mode='update')
        self.register_sampler(name='high_price', mode='max')
        self.register_sampler(name='low_price', mode='min')
        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='trade_flow', mode='accumulate')

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.beta.{ticker}' for ticker in subscription
        ] + [
            f'{self.name.removeprefix("Monitor.")}.r2.{ticker}' for ticker in subscription
        ]

    def _param_range(self) -> dict[str, list[...]]:
        # give some simple parameter range
        params_range = super()._param_range()

        params_range.update(
            sampling_interval=[15, 30, 60],
            sample_size=[10, 20, 30]
        )

        return params_range

    @property
    def value(self) -> dict[str, float]:
        factor_value = {}

        qrs_result = self.qrs()

        for ticker, qrs_analysis in qrs_result.items():
            beta, r2 = qrs_analysis
            factor_value[f'beta.{ticker}'] = beta
            factor_value[f'r2.{ticker}'] = r2

        return factor_value


class MomentumSupportAdaptiveMonitor(MomentumSupportMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = False, name: str = 'Monitor.QRS.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval
        )

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        self.accumulate_volume(market_data=trade_data)
        super().on_trade_data(trade_data=trade_data, **kwargs)

    @property
    def is_ready(self) -> bool:
        return self.baseline_stable


class MomentumSupportIndexMonitor(MomentumSupportMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float] = None, name: str = 'Monitor.QRS.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Beta',
            f'{self.name.removeprefix("Monitor.")}.Beta.Slope',
            f'{self.name.removeprefix("Monitor.")}.R2'
        ]

    @property
    def value(self) -> dict[str, float]:
        qrs_analysis = self.qrs()
        beta_dict = {}
        r2_dict = {}

        for ticker, (slope, r_squared) in qrs_analysis.items():
            beta_dict[ticker] = slope
            r2_dict[ticker] = r_squared

        return {
            'Beta': self.composite(values=beta_dict),
            'Beta.Slope': self.composite(values=self.slope(key='beta')),
            'R2': self.composite(values=r2_dict)
        }


class MomentumSupportAdaptiveIndexMonitor(MomentumSupportAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.QRS.Adaptive.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Beta',
            f'{self.name.removeprefix("Monitor.")}.Beta.Slope',
            f'{self.name.removeprefix("Monitor.")}.R2'
        ]

    @property
    def value(self) -> dict[str, float]:
        qrs_analysis = self.qrs()
        beta_dict = {}
        r2_dict = {}

        for ticker, (slope, r_squared) in qrs_analysis.items():
            beta_dict[ticker] = slope
            r2_dict[ticker] = r_squared

        return {
            'Beta': self.composite(values=beta_dict),
            'Beta.Slope': self.composite(values=self.slope(key='beta')),
            'R2': self.composite(values=r2_dict)
        }

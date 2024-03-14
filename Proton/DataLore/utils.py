from typing import Iterable

import numpy as np
import pandas as pd


class Panel(object):
    """
    for each stock, we collect a panel data.
    """

    def __init__(self, ticker: str, sampling_interval: float):
        self.ticker = ticker
        self.sampling_interval = sampling_interval
        self.index = []
        self.features = []
        self.price: dict[int, float] = {}
        self.factor: dict[int, dict[str, float]] = {}

        self.last_index = 0

    def observation(self, factor_value: dict[str, float], market_price: float, timestamp: float = None):
        # without timestamp, just log the observation with self-addressing index
        if timestamp is None:
            idx = self.last_index + 1
        else:
            idx = int(timestamp // self.sampling_interval)

        # update index
        if idx != self.last_index:
            self.last_index = idx
            self.index.append(idx)

        self.features += [feature_name for feature_name in factor_value if feature_name not in self.features]
        self.price[idx] = market_price
        self.factor[idx] = factor_value

    def define_inputs(self, factor_value: pd.DataFrame | dict[str, float | int] = None, input_vars: Iterable[str] = None, poly_degree: int = 1, timestamp: float | list[float] = None) -> pd.DataFrame:
        if factor_value is None:
            factor_value = pd.DataFrame(self.factor).T
            factor_value.index = [_ * self.sampling_interval for _ in self.index]

        if input_vars is None:
            input_vars = self.features

        from Quark.DataLore.utils import define_inputs

        x_matrix = define_inputs(
            factor_value=factor_value,
            input_vars=input_vars,
            poly_degree=poly_degree,
            timestamp=timestamp
        )

        return x_matrix

    def define_prediction(self, pred_length: float, session_filter: callable = None) -> pd.Series:
        target = {}

        price_series = {idx * self.sampling_interval: self.price[idx] for idx in self.price}

        for ts, px in price_series.items():
            t0 = ts
            t1 = t0 + pred_length

            if session_filter is not None and not session_filter(t0):
                continue

            closest_index = None

            for index in price_series.keys():
                if index >= t1:
                    closest_index = index
                    break

            if closest_index is None:
                continue

            # Get the prices at ts and ts + window
            p0 = px
            p1 = price_series[closest_index]

            # Calculate the percentage change and assign it to the 'pct_chg' column
            target[t0] = (p1 / p0) - 1

        return pd.Series(target)

    def clear(self):
        self.index.clear()
        self.features.clear()
        self.price.clear()
        self.factor.clear()
        self.last_index = 0


class Scaler(object):
    def __init__(self):
        self.scaler: pd.DataFrame | None = None

    def standardization_scaler(self, x: pd.DataFrame):
        scaler = pd.DataFrame(index=['mean', 'std'], columns=x.columns)

        for col in x.columns:
            if col == 'Bias':
                scaler.loc['mean', col] = 0
                scaler.loc['std', col] = 1
            else:
                valid_values = x[col][np.isfinite(x[col])]
                scaler.loc['mean', col] = np.mean(valid_values)
                scaler.loc['std', col] = np.std(valid_values)

        self.scaler = scaler
        return scaler

    def transform(self, x: pd.DataFrame | dict[str, float]) -> pd.DataFrame | dict[str, float]:
        if self.scaler is None:
            raise ValueError('scaler not initialized!')

        if isinstance(x, pd.DataFrame):
            x = (x - self.scaler.loc['mean']) / self.scaler.loc['std']
        elif isinstance(x, dict):
            for var_name in x:

                if var_name not in self.scaler.columns:
                    # LOGGER.warning(f'{var_name} is not in scaler')
                    continue

                x[var_name] = (x[var_name] - self.scaler.at['mean', var_name]) / self.scaler.at['std', var_name]
        else:
            raise TypeError(f'Invalid x type {type(x)}, expect dict or pd.DataFrame')

        return x


__all__ = ['Panel']

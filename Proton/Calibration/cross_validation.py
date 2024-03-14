import pathlib
from functools import cached_property
from typing import Hashable

import numpy as np
import pandas as pd

from . import Regression, LOGGER
from .Boosting import Boosting
from .Linear import LinearBootstrap, LinearRegression
from .kelly import kelly_bootstrap
from ..Base import GlobalStatics
from .metrics import Metrics

RANGE_BREAK = GlobalStatics.RANGE_BREAK


class CrossValidation(object):
    def __init__(self, model: Regression, folds=5, shuffle: bool = True, trade_cost: float = 0.0016, **kwargs):
        """
        Initialize the CrossValidation object.

        Args:
            model: Regression model object (e.g., LinearRegression).
            folds (int): Number of folds for cross-validation.
            strict_no_future (bool): training data must be prier to ALL the validation data
            shuffle (bool): shuffle the training and validation index
        """
        self.model = model
        self.folds = folds
        self.shuffle = shuffle
        self.trade_cost = trade_cost

        self.fit_kwargs = kwargs

        self.x_val = None
        self.y_val = None
        self.y_pred = None
        self.prediction_interval = None
        self.resampled_deviation = None

        self._metrics = None
        self._fig = None

    @classmethod
    def _select_data(cls, x: np.ndarray, y: np.ndarray, indices: np.ndarray, fold: int, n_folds: int, shuffle: bool = False):
        n = len(x)
        start_idx = (n // n_folds) * fold
        end_idx = (n // n_folds) * (fold + 1) if fold < n_folds - 1 else n

        val_indices = indices[start_idx:end_idx].copy()
        train_indices = np.setdiff1d(indices, val_indices).copy()

        if shuffle:
            np.random.shuffle(val_indices)
            np.random.shuffle(train_indices)

        x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]

        return x_train, y_train, x_val, y_val, val_indices

    def _predict(self, x):
        if isinstance(self.model, LinearBootstrap):
            y_pred, prediction_interval, resampled_deviation, *_ = self.model.predict(x=x)
        elif isinstance(self.model, Boosting):
            y_pred, prediction_interval, *_ = self.model.predict(x=x)
            resampled_y = self.model.resample(x=x)
            resampled_deviation = []
            for _y, _y_resampled in zip(y_pred, resampled_y):
                resampled_deviation.append(_y_resampled - _y)
            resampled_deviation = np.array(resampled_deviation)
        else:
            raise NotImplementedError(f'Can not find validation method for {self.model.__class__}')

        return y_pred, prediction_interval, resampled_deviation

    def cross_validate(self, x: np.ndarray, y: np.ndarray):
        """
        Perform cross-validation and store the results in the metrics attribute.

        Args:
            x (numpy.ndarray): Input features.
            y (numpy.ndarray): Output values.

        Returns:
            None
        """
        n = len(x)
        indices = np.arange(n)
        np.random.shuffle(indices)

        fold_metrics = {'x_val': [], 'y_val': [], 'y_pred': [], 'index': [], 'prediction_interval': [], 'resampled_deviation': []}

        for fold in range(self.folds):
            x_train, y_train, x_val, y_val, val_indices = self._select_data(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)
            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

            # Predict on the validation data
            y_pred, prediction_interval, resampled_deviation = self._predict(x_val)

            # this type model will accumulate bootstrap instances on each fit.
            if isinstance(self.model, LinearRegression):
                resampled_deviation = resampled_deviation[:, -self.model.bootstrap_samples:]

            fold_metrics['x_val'].append(x_val)
            fold_metrics['y_val'].append(y_val)
            fold_metrics['y_pred'].append(y_pred)
            fold_metrics['index'].append(val_indices)  # Store sorted indices
            fold_metrics['prediction_interval'].append(prediction_interval)
            fold_metrics['resampled_deviation'].append(resampled_deviation)

        sorted_indices = np.concatenate(fold_metrics['index'])
        for key in ['x_val', 'y_val', 'y_pred', 'prediction_interval', 'resampled_deviation']:
            values = np.concatenate(fold_metrics[key])
            fold_metrics[key] = values[np.argsort(sorted_indices)]

        self.x_val = fold_metrics['x_val']
        self.y_val = fold_metrics['y_val']
        self.y_pred = fold_metrics['y_pred']
        self.prediction_interval = fold_metrics['prediction_interval']
        self.resampled_deviation = fold_metrics['resampled_deviation']

    def validate(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, skip_fitting: bool = False):
        if not skip_fitting:
            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

        # Predict on the validation data
        y_pred, prediction_interval, resampled_deviation = self._predict(x_val)

        self.x_val = np.array(x_val)
        self.y_val = np.array(y_val)
        self.y_pred = np.array(y_pred)
        self.prediction_interval = np.array(prediction_interval)
        self.resampled_deviation = np.array(resampled_deviation)

    def plot(self, **kwargs):
        # todo: not implemented
        if self._fig is not None:
            return self._fig

        import plotly.graph_objects as go

        fig = go.Figure()
        self._fig = fig
        return fig

    def clear(self):
        self.x_val = None
        self.y_val = None
        self.y_pred = None
        self.prediction_interval = None

        self._metrics = None
        self._fig = None

    @property
    def metrics(self) -> Metrics | None:
        if self.x_val is None or self.y_val is None:
            raise ValueError('Must call .validation() method first')

        if self._metrics is None:
            self._metrics = Metrics(
                y_true=self.y_val,
                y_val=self.y_pred,
                x_val=self.x_val,
                model=self.model,
                alpha=0.10,
                trade_cost=self.trade_cost
            )

        return self._metrics

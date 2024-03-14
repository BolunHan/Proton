"""markdown
This module is designed to handel and calculate metrics for stock selection algos.

# Design of the Metrics

All metrics should be calculated using out-sample validation (Required a fitted model). The class is designed to calculate cross-sectional data (not panel data), y_val and y_true is supposed to be a 1-d array. x_val can be n-d array (multidimensional factor or factor pool).

y_val and y_true should be passed in explicitly during the initialization of the class.
- If the x_val (which normally is a vector of factor) is a 1-d array, then the x_val -> y_val is a linear transformation. Most of the metrics are resilient to the linear transformation, so model parameter can be omitted.
- If y_val is not provided, then y_val is calculated with model.predict(x_val)
- Some metrics requires a bootstrap model, e.g. kelly related ones. The prediction interval might be calculated multiple times. A caching function is enabled by default.


# Metrics Function

1. Metrics of Error
- MAE
- MSE
2. Metrics of Accuracy
- Acc
- Acc_Quantile
3. Metrics of Information
- IC
- IC_Pearson
- IC_Quantile
- IC_Pearson_Quantile
4. Metrics of Portfolio
- IR: Note that this requires panel data, IR can only be calculated sequentially during the backtest. Also, IR assumes equal weights for each selected stocks. The Metrics provides a function to calculate portfolio return.
- IR_Quantile
- Kelly
5. AUC_ROC
- Acc_AUC_ROC
- IC_AUC_ROC

# Usage of Metrics

Prototyping: use IC_Pearson (fast, no need of training)
Minimizing Risk: use Acc
Maximizing Gain: use IR, Kelly
Factor Ensemble: use IR

If only one metrics is used to select factor, I would recommend using AUC_ROC of IR.
"""

import pathlib
from functools import cached_property
from typing import Hashable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import Regression, LOGGER
from .Linear import LinearBootstrap
from .kelly import kelly_bootstrap
from ..Base import GlobalStatics

RANGE_BREAK = GlobalStatics.RANGE_BREAK


class Cache(object):
    """
    Cache decorator for memoization.
    """

    def __init__(self):
        self.cache = {}

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            entry_list = []

            for _ in args:
                if isinstance(_, Hashable):
                    entry_list.append(_)
                else:
                    entry_list.append(id(_))

            for _ in kwargs.values():
                if isinstance(_, Hashable):
                    entry_list.append(_)
                else:
                    entry_list.append(id(_))

            cache_key = tuple(entry_list)

            if cache_key in self.cache:
                value = self.cache[cache_key]
            else:
                value = self.cache[cache_key] = func(*args, **kwargs)

            return value

        return wrapper


METRIC_CACHE = Cache()


class Metrics(object):
    """
    Metrics class for evaluating regression model performance.

    y_true must be the actual return of the underlying (so that the kelly can makes sense)
    """

    def __init__(self, y_true: np.ndarray, x_val: np.ndarray = None, model: Regression = None, y_val: np.ndarray = None, alpha=0.10, trade_cost=0.0016):
        self.model: Regression = model
        self.x_val: np.ndarray = x_val
        self.y_val: np.ndarray = y_val  # must assign a y_val or a model
        self.y_true: np.ndarray = y_true

        if x_val is None and y_val is None:
            LOGGER.warning('Must assign a x_val or y_val')
        elif y_val is None:
            if model is None and len(x_val.shape) == 1:
                self.y_val = x_val  # Since the model is typically a linear one. Most metrics is not sensitive to the linear transformation. This proxy is acceptable in most of the cases.
            elif model is not None:
                self.y_val, *_ = model.predict(x_val)

        self.alpha = alpha
        self.alpha_range = np.linspace(0., 1, 100)
        self.trade_cost = trade_cost

    def __del__(self):
        """
        Destructor to clean up the object.
        """
        self.model = None
        self.x_val = None
        self.y_val = None
        self.y_true = None
        METRIC_CACHE.cache.clear()

    @classmethod
    @METRIC_CACHE
    def _predict(cls, model: Regression, x: np.ndarray, alpha: float):
        """
        Predict using the regression model.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            alpha (float): Significance level for the prediction interval.

        Returns:
            Tuple: Predicted values, prediction interval, and residuals.
        """
        return model.predict(x=x, alpha=alpha)

    @classmethod
    def _compute_kelly(cls, model: Regression, x_val: np.ndarray, max_leverage: float = 2., cost: float = 0.0016):
        y_pred, _, bootstrap_deviation, *_ = cls._predict(model=model, x=x_val, alpha=0)

        kelly_value = []
        for outcome, deviations in zip(y_pred, bootstrap_deviation):
            kelly_proportion = kelly_bootstrap(outcomes=np.array(deviations) + outcome, cost=cost, max_leverage=max_leverage)
            kelly_value.append(kelly_proportion)

        return np.array(kelly_value)

    # --- some basic ones (for design of advanced cost function) ---
    @classmethod
    def _metric_mae(cls, y_true: np.ndarray, y_val: np.ndarray) -> float:
        residuals = y_true - y_val
        mae = np.mean(np.abs(residuals))
        return mae

    @classmethod
    def _metric_mse(cls, y_true: np.ndarray, y_val: np.ndarray) -> float:
        residuals = y_true - y_val
        mse = np.mean(residuals ** 2)
        return mse

    @classmethod
    def _metric_accuracy(cls, y_true: np.ndarray, y_val: np.ndarray) -> float:
        correct_signs = np.sign(y_true) == np.sign(y_val)
        accuracy = np.mean(correct_signs)
        return accuracy

    @classmethod
    def _metric_ic(cls, y_true: np.ndarray, y_val: np.ndarray):
        """
        no need to remove the std of y_true. the amplitude of pct_return serves as a natural weights of this metrics.
        this will not affect factor selection (same weight when using cross-sectional data).
        however the ic prefers strong linear relationship (unfairly under-weights the non-linear factor, which is common when selecting top / bottom pred quantile).
        use Person corr to address this issue.
        """
        covariance_matrix = np.cov(y_val, y_true)

        variance, covariance = covariance_matrix[0]

        if not variance or not np.isfinite(variance):
            return np.nan

        ic = covariance / variance
        return ic

    @classmethod
    def _metric_ic_person(cls, y_true: np.ndarray, y_val: np.ndarray):
        """
        no need to remove the std of y_true. the amplitude of pct_return serves as a natural weights of this metrics.
        this will not affect factor selection (same weight when using cross-sectional data).
        however the ic prefers strong linear relationship (unfairly under-weights the non-linear factor, which is common when selecting top / bottom pred quantile).
        use Person corr to address this issue.
        """
        covariance_matrix = np.cov(rankdata(y_val), rankdata(y_true))

        variance, covariance = covariance_matrix[0]

        if not variance or not np.isfinite(variance):
            return np.nan

        ic = covariance / variance
        return ic

    # --- variants of the metrics ---
    @classmethod
    def _select_quantile(cls, y_val: np.ndarray, y_true: np.ndarray, quantile: float, side: str = 'both') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        for side of up and down, the quantile value passed in is not exactly the actual quantile (half of it)
        this feature is retained so auc-roc can be calculated without much of treadmill
        """
        num_samples = len(y_val)
        num_quantile_samples = int(num_samples * quantile / 2)  # this round down will avoid quantile crossing

        # Sort y_val indices based on values
        sorted_indices = np.argsort(y_val)

        # Select top and bottom quantile samples
        top_quantile_indices = sorted_indices[-num_quantile_samples:]
        bottom_quantile_indices = sorted_indices[:num_quantile_samples]

        if side == 'both':
            quantile_indices = np.concatenate((top_quantile_indices, bottom_quantile_indices))
        elif side == 'up':
            quantile_indices = top_quantile_indices
        elif side == 'down':
            quantile_indices = bottom_quantile_indices
        else:
            raise ValueError(f'Invalid side {side}, expect "up", "down" or "both"')

        return y_true[quantile_indices], y_val[quantile_indices], quantile_indices

    @classmethod
    def _metric_accuracy_quantile(cls, y_true: np.ndarray, y_val: np.ndarray, quantile: float = 1., side: str = 'both') -> float:
        y_true_selected, y_val_selected, indices_selected = cls._select_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        return cls._metric_accuracy(y_true=y_true_selected, y_val=y_val_selected)

    @classmethod
    def _metric_ic_quantile(cls, y_true: np.ndarray, y_val: np.ndarray, quantile: float = 1., side: str = 'both') -> float:
        y_true_selected, y_val_selected, indices_selected = cls._select_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        return cls._metric_ic(y_true=y_true_selected, y_val=y_val_selected)

    @classmethod
    def _metric_ic_pearson_quantile(cls, y_true: np.ndarray, y_val: np.ndarray, quantile: float = 1., side: str = 'both') -> float:
        y_true_selected, y_val_selected, indices_selected = cls._select_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        return cls._metric_ic_person(y_true=y_true_selected, y_val=y_val_selected)

    # --- metrics for portfolio building ---
    @classmethod
    def _metric_alpha(cls, y_true: np.ndarray, y_val: np.ndarray, quantile: float = 1., side: str = 'both', trade_cost: float = 0.) -> float:
        y_true_selected, y_val_selected, indices_selected = cls._select_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        weights = []

        for y_pred in y_val_selected:
            if y_pred > trade_cost and (side == 'up' or side == 'both'):
                weights.append(1)
            elif y_pred < -trade_cost and (side == 'down' or side == 'both'):
                weights.append(-1)
            else:
                weights.append(0)

        baseline = np.mean(y_true)  # should not use y_true_selected, this is for benchmark
        alpha = (y_true_selected - baseline)
        portfolio_alpha = np.average(alpha, weights=weights)

        return portfolio_alpha

    @classmethod
    def _metric_alpha_kelly(cls, y_true: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, bootstrap_model: LinearBootstrap, quantile: float = 1., side: str = 'both', max_leverage: float = 2., trade_cost: float = 0.) -> float:
        y_true_selected, y_val_selected, indices_selected = cls._select_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        kelly_leverage = cls._compute_kelly(model=bootstrap_model, x_val=x_val, max_leverage=max_leverage, cost=trade_cost)[indices_selected]
        weights = []

        for leverage in kelly_leverage:
            if leverage > 0 and (side == 'up' or side == 'both'):
                weights.append(leverage)
            elif leverage < 0 and (side == 'down' or side == 'both'):
                weights.append(leverage)
            else:
                weights.append(0)

        baseline = np.mean(y_true)
        alpha = (y_true_selected - baseline)
        portfolio_alpha = np.average(alpha, weights=kelly_leverage)

        return portfolio_alpha

    # --- AUC-ROC of Metrics ---
    @classmethod
    def _calculate_quantile_aucroc(cls, metric_function: callable, quantile_range: np.ndarray = np.linspace(0., 1, 100), *args, **kwargs):
        roc_values = []
        for quantile in quantile_range:
            accuracy = metric_function(quantile=quantile, *args, **kwargs)
            roc_values.append(accuracy)

        roc_values = np.array(roc_values)
        valid_indices = ~np.isnan(roc_values)

        if np.any(valid_indices):
            auc_roc = -np.trapz(roc_values[valid_indices], (1 - quantile_range)[valid_indices])
            return auc_roc
        else:
            return np.nan

    @classmethod
    def _metric_accuracy_aucroc(cls, y_true: np.ndarray, y_val: np.ndarray, quantile_range: np.ndarray = np.linspace(0., 1, 100), side: str = 'both'):
        # accuracy = cls._metric_accuracy_quantile(y_true=y_true, y_val=y_val, quantile=1., side=side)
        auc_roc = cls._calculate_quantile_aucroc(cls._metric_accuracy_quantile, quantile_range=quantile_range, y_true=y_true, y_val=y_val, side=side)
        return auc_roc

    @classmethod
    def _metric_ic_aucroc(cls, y_true: np.ndarray, y_val: np.ndarray, quantile_range: np.ndarray = np.linspace(0., 1, 100), side: str = 'both'):
        # ic = cls._metric_ic_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        auc_roc = cls._calculate_quantile_aucroc(cls._metric_ic_quantile, quantile_range=quantile_range, y_true=y_true, y_val=y_val, side=side)
        return auc_roc

    @classmethod
    def _metric_ic_pearson_aucroc(cls, y_true: np.ndarray, y_val: np.ndarray, quantile_range: np.ndarray = np.linspace(0., 1, 100), side: str = 'both'):
        # ic_rank = cls._metric_ic_pearson_quantile(y_true=y_true, y_val=y_val, quantile=quantile, side=side)
        auc_roc = cls._calculate_quantile_aucroc(cls._metric_ic_pearson_quantile, quantile_range=quantile_range, y_true=y_true, y_val=y_val, side=side)
        return auc_roc

    @classmethod
    def _metric_alpha_aucroc(cls, y_true: np.ndarray, y_val: np.ndarray, quantile_range: np.ndarray = np.linspace(0., 1, 100), side: str = 'both', trade_cost: float = 0.):
        # alpha = cls._metric_alpha(y_true=y_true, y_val=y_val, quantile=1., side=side, trade_cost=trade_cost)
        auc_roc = cls._calculate_quantile_aucroc(cls._metric_alpha, quantile_range=quantile_range, y_true=y_true, y_val=y_val, side=side, trade_cost=trade_cost)
        return auc_roc

    @classmethod
    def _metric_kelly_aucroc(cls, y_true: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, bootstrap_model: LinearBootstrap, quantile_range: np.ndarray = np.linspace(0., 1, 100), max_leverage: float = 2., side: str = 'both', trade_cost: float = 0.):
        # alpha = cls._metric_alpha_kelly(y_true=y_true, x_val=x_val, y_val=y_val, bootstrap_model=model, quantile=1., side=side, max_leverage=max_leverage, trade_cost=trade_cost)
        auc_roc = cls._calculate_quantile_aucroc(cls._metric_alpha_kelly, quantile_range=quantile_range, y_true=y_true, x_val=x_val, y_val=y_val, bootstrap_model=bootstrap_model, side=side, max_leverage=max_leverage, trade_cost=trade_cost)
        return auc_roc

    @classmethod
    def plot_roc(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha_range: np.ndarray, accuracy_baseline: float = 0., **kwargs):
        # todo: need update
        import plotly.graph_objects as go

        roc_values = []
        selection_ratios = []

        for alpha in alpha_range:
            accuracy, selection_ratio = cls.compute_accuracy_significant(model=model, x=x, y_actual=y_actual, alpha=alpha)
            roc_values.append(accuracy - accuracy_baseline)
            selection_ratios.append(selection_ratio)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=1 - alpha_range,
                y=roc_values,
                mode='lines',
                name=kwargs.get('curve_name', "ROC Curve"),
                line=dict(color='blue'),
                yaxis='y1'
            )
        )

        fig.add_trace(
            go.Bar(
                x=1 - alpha_range,
                y=selection_ratios,
                opacity=0.3,
                name="Selection Ratio",
                marker=dict(color='green'),
                yaxis='y2'
            )
        )

        fig.update_layout(
            title=kwargs.get('title', "Receiver Operating Characteristic (ROC) Curve"),
            xaxis_title=kwargs.get('x_name', "Classification threshold (1 - alpha)"),
            yaxis_title=kwargs.get('y_name', "Accuracy Boost" if accuracy_baseline else "Accuracy"),
            hovermode="x unified",
            template='simple_white',
            showlegend=False,
            yaxis=dict(
                maxallowed=0.5 if accuracy_baseline else 1,
                minallowed=-0.5 if accuracy_baseline else 0,
                showspikes=True,
                tickformat='.2%'
            ),
            yaxis2=dict(
                title="Selection Ratio",
                range=[0, 1],
                overlaying='y',
                side='right',
                showgrid=False,
                tickformat='.2%'
            )
        )

        return fig

    def to_html(self, file_path: str | pathlib.Path):
        # todo: need update
        metrics_data = self.metrics.copy()

        # Create metrics table figure
        # noinspection PyTypeChecker
        metrics_data['obs_num'] = f"{metrics_data['obs_num']:,d}"
        metrics_table = pd.DataFrame({'Metrics': pd.Series(metrics_data)})

        # Create ROC curve figure
        roc_curve = self.plot_roc(
            model=self.model,
            x=self.x,
            y_actual=self.y,
            accuracy_baseline=metrics_data['accuracy_baseline'],
            alpha_range=self.alpha_range
        )

        # Convert the figures to HTML codes
        metrics_table_html = metrics_table.to_html(float_format=lambda x: f'{x:.4%}')
        roc_curve_html = roc_curve.to_html(roc_curve, full_html=False)

        # Create a 1x2 table HTML code
        html_code = f"""
        <html>
        <head></head>
        <body>
            <table style="width:100%">
                <tr>
                    <td style="width:30%">{metrics_table_html}</td>
                    <td style="width:70%">{roc_curve_html}</td>
                </tr>
            </table>
        </body>
        </html>
        """

        # Write the HTML code to the file
        with open(file_path, 'w') as file:
            file.write(html_code)

    @cached_property
    def metrics(self) -> dict[str, int | float]:
        metrics = dict(
            obs_num=len(self.x_val),
            quantile_default=self.alpha,
            quantile_range_from=self.alpha_range[0],
            quantile_range_to=self.alpha_range[-1],
            trade_cost=self.trade_cost
        )

        # this should not happen, the initialization have assertion to ensure that.
        if self.x_val is None and self.y_val is None:
            LOGGER.error('Can not calculate metrics, must assign a y_val or x_val.')
            return metrics

        if self.y_val is not None:
            # Basic Error
            metrics['MSE'] = self._metric_mse(y_true=self.y_true, y_val=self.y_val)
            metrics['MAE'] = self._metric_mae(y_true=self.y_true, y_val=self.y_val)

            # Acc Metric
            metrics['Acc'] = self._metric_accuracy(y_true=self.y_true, y_val=self.y_val)
            metrics['Acc_Up'] = self._metric_accuracy_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='up')
            metrics['Acc_Down'] = self._metric_accuracy_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='down')

            metrics['Acc_Selected'] = self._metric_accuracy_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha)
            metrics['Acc_Up_Selected'] = self._metric_accuracy_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='up')
            metrics['Acc_Down_Selected'] = self._metric_accuracy_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='down')

            metrics['Acc_AUCROC'] = self._metric_accuracy_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range)
            metrics['Acc_Up_AUCROC'] = self._metric_accuracy_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='up')
            metrics['Acc_Down_AUCROC'] = self._metric_accuracy_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='down')

            # IC Metric
            metrics['IC'] = self._metric_ic(y_true=self.y_true, y_val=self.y_val)
            metrics['IC_Up'] = self._metric_ic_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='up')
            metrics['IC_Down'] = self._metric_ic_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='down')

            metrics['IC_Selected'] = self._metric_ic_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha)
            metrics['IC_Up_Selected'] = self._metric_ic_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='up')
            metrics['IC_Down_Selected'] = self._metric_ic_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='down')

            metrics['IC_AUCROC'] = self._metric_ic_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range)
            metrics['IC_Up_AUCROC'] = self._metric_ic_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='up')
            metrics['IC_Down_AUCROC'] = self._metric_ic_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='down')

            # Rank IC Metric
            metrics['IC_Pearson'] = self._metric_ic_person(y_true=self.y_true, y_val=self.y_val)
            metrics['IC_Pearson_Up'] = self._metric_ic_pearson_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='up')
            metrics['IC_Pearson_Down'] = self._metric_ic_pearson_quantile(y_true=self.y_true, y_val=self.y_val, quantile=1., side='down')

            metrics['IC_Pearson_Selected'] = self._metric_ic_pearson_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha)
            metrics['IC_Pearson_Up_Selected'] = self._metric_ic_pearson_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='up')
            metrics['IC_Pearson_Down_Selected'] = self._metric_ic_pearson_quantile(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, side='down')

            metrics['IC_Pearson_AUCROC'] = self._metric_ic_pearson_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range)
            metrics['IC_Pearson_Up_AUCROC'] = self._metric_ic_pearson_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='up')
            metrics['IC_Pearson_Down_AUCROC'] = self._metric_ic_pearson_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, side='down')

            # Alpha Metric
            metrics['Alpha'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=1., trade_cost=self.trade_cost)
            metrics['Alpha_Up'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=1., trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Down'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=1., trade_cost=self.trade_cost, side='down')

            metrics['Alpha_Selected'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, trade_cost=self.trade_cost)
            metrics['Alpha_Up_Selected'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Down_Selected'] = self._metric_alpha(y_true=self.y_true, y_val=self.y_val, quantile=self.alpha, trade_cost=self.trade_cost, side='down')

            metrics['Alpha_AUCROC'] = self._metric_alpha_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, trade_cost=self.trade_cost)
            metrics['Alpha_Up_AUCROC'] = self._metric_alpha_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Down_AUCROC'] = self._metric_alpha_aucroc(y_true=self.y_true, y_val=self.y_val, quantile_range=self.alpha_range, trade_cost=self.trade_cost, side='down')

        if self.y_val is not None and self.x_val is not None and isinstance(self.model, LinearBootstrap):
            # Kelly Alpha Metric
            metrics['Alpha_Kelly'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=1., trade_cost=self.trade_cost)
            metrics['Alpha_Kelly_Up'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=1., trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Kelly_Down'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=1., trade_cost=self.trade_cost, side='down')

            metrics['Alpha_Kelly_Selected'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=self.alpha, trade_cost=self.trade_cost)
            metrics['Alpha_Kelly_Up_Selected'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=self.alpha, trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Kelly_Down_Selected'] = self._metric_alpha_kelly(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile=self.alpha, trade_cost=self.trade_cost, side='down')

            metrics['Alpha_Kelly_AUCROC'] = self._metric_kelly_aucroc(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile_range=self.alpha_range, trade_cost=self.trade_cost)
            metrics['Alpha_Kelly_Up_AUCROC'] = self._metric_kelly_aucroc(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile_range=self.alpha_range, trade_cost=self.trade_cost, side='up')
            metrics['Alpha_Kelly_Down_AUCROC'] = self._metric_kelly_aucroc(y_true=self.y_true, x_val=self.x_val, y_val=self.y_val, bootstrap_model=self.model, quantile_range=self.alpha_range, trade_cost=self.trade_cost, side='down')

        return metrics

"""
This script is designed for factor validation using linear regression.
"""
__package__ = 'Proton.Factor'

import datetime
import os
import pathlib
import random
from typing import Iterable

import numpy as np
import pandas as pd
from AlgoEngine.Engine import ProgressiveReplay
from PyQuantKit import MarketData
from Quark.Backtest import simulated_env
from Quark.Factor.Misc import SyntheticIndexMonitor
from Quark.Misc import helper
from Quark.Profile import cn

from . import LOGGER, MDS
from .utils import IndexWeight, FactorMonitor, MONITOR_MANAGER
from ..API import historical
from ..Base import safe_exit, GlobalStatics
from ..Calibration.Linear.bootstrap import *
from ..Calibration.cross_validation import CrossValidation
from ..Calibration.dummies import is_market_session
from ..DataLore.utils import Panel

cn.profile_cn_override()
LOGGER = LOGGER.getChild('validation')
DUMMY_WEIGHT = True
TIME_ZONE = GlobalStatics.TIME_ZONE
RANGE_BREAK = GlobalStatics.RANGE_BREAK
START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 4, 1)
MDS.monitor_manager = MONITOR_MANAGER
INDEX_WEIGHTS = IndexWeight(index_name='000016.SH')


class FactorValidation(object):
    def __init__(self, **kwargs):
        # Params for replay
        self.dtype = kwargs.get('dtype', 'TradeData')
        self.start_date = kwargs.get('start_date', START_DATE)
        self.end_date = kwargs.get('end_date', END_DATE)

        # Params for index
        self.index_name = kwargs.get('index_name', '000016.SH')
        self.index_weights = INDEX_WEIGHTS
        self._update_index_weights(market_date=self.start_date)
        self.subscription = list(self.index_weights.keys())

        # Params for sampling
        self.sampling_interval = kwargs.get('sampling_interval', 5 * 60.)

        # Params for validation
        self.poly_degree = kwargs.get('poly_degree', 1)
        self.pred_target = 60 * 60  # 3600 = 1hr, 7200 = 2hr, etc...

        self.factor: list[FactorMonitor] = factor if isinstance(factor := kwargs.get('factor', []), list) else [factor]
        self.synthetic = SyntheticIndexMonitor(index_name=self.index_name, weights=self.index_weights, interval=self.sampling_interval)
        self.observation: dict[str, Panel] = {}
        self.last_idx = 0

        self.model = RidgeRegression(alpha=.2, exponential_decay=0.25, fixed_decay=0.5)
        self.cv = CrossValidation(model=self.model, folds=10, shuffle=True)
        self.metrics = {}
        self.validation_id = kwargs.get('validation_id', self._get_validation_id())

    def _get_validation_id(self):
        validation_id = 1

        while True:
            dump_dir = f'{self.__class__.__name__}.{validation_id}'
            if os.path.isdir(dump_dir):
                validation_id += 1
            else:
                break

        return validation_id

    def _collect_factor(self):
        factor_value = MDS.monitor_manager.values

        for ticker in self.index_weights:
            _factor = {}
            _price = MDS.market_price.get(ticker, None)

            if _price is None:
                continue

            if ticker in self.observation:
                _panel = self.observation[ticker]
            else:
                _panel = self.observation[ticker] = Panel(ticker=ticker, sampling_interval=self.sampling_interval)

            for name, value in factor_value.items():  # type: str, float
                if name.endswith(ticker):
                    _factor[name.rstrip(ticker)] = value

            _panel.observation(
                factor_value=_factor,
                market_price=_price,
                timestamp=MDS.timestamp
            )

    def _cross_validation(self, x, y, factors: pd.DataFrame, cv: CrossValidation):
        valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        x_axis = np.array([datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factors.index])[valid_mask]

        cv.cross_validate(x=x, y=y)
        cv.x_axis = x_axis

    def _update_index_weights(self, market_date: datetime.date):
        """
        Updates index weights based on the provided market date.

        Args:
            market_date (datetime.date): Date for which to update index weights.
        """
        index_weights = IndexWeight(
            index_name=self.index_name,
            **helper.load_dict(
                file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{self.index_name}.{market_date:%Y%m%d}.json'),
                json_dict=simulated_env.query(ticker=self.index_name, market_date=market_date, topic='index_weights')
            )
        )

        # A lite setting for fast debugging
        if DUMMY_WEIGHT:
            for _ in list(index_weights.keys())[5:]:
                index_weights.pop(_)

        # Step 0: Update index weights
        self.index_weights.clear()
        self.index_weights.update(index_weights)
        self.index_weights.normalize()
        # the factors and synthetics are using the same index_weights reference, so no need to update individually
        # self.synthetic.weights = self.index_weights
        return index_weights

    def _update_subscription(self, replay: ProgressiveReplay):
        """
        Updates market data subscriptions based on index weights.
        """
        self.subscription.clear()
        replay.replay_subscription.clear()

        subscription = set(self.index_weights.keys())
        self.subscription.extend(subscription)

        if isinstance(self.dtype, str):
            dtype = [self.dtype]
        elif isinstance(self.dtype, Iterable):
            dtype = list(self.dtype)
        else:
            raise ValueError(f'Invalid dtype {self.dtype}')

        for ticker in subscription:
            for _dtype in dtype:
                replay.add_subscription(ticker=ticker, dtype=_dtype)

        # this should be only affecting behaviors of the concurrent manager using the shm feature.
        MDS.monitor_manager.subscription = subscription

    def initialize_factor(self, **kwargs) -> list[FactorMonitor]:
        """
        must re-initialize factor on each day, this ensures GC.
        """
        factors = []

        for factor in self.factor:
            _factor = factor.__class__(**factor.params)
            factors.append(_factor)

        self.factor.clear()
        self.factor.extend(factors)

        for factor in self.factor:
            MDS.add_monitor(factor)

        MDS.add_monitor(self.synthetic)

        return self.factor

    def validation(self, market_date: datetime.date):
        x = []
        y = []

        for ticker, panel in self.observation.items():
            _x = panel.define_inputs(poly_degree=self.poly_degree)
            _y = panel.define_prediction(pred_length=self.pred_target)

            _x['pred_target'] = _y
            _x.dropna(axis=0, inplace=True)
            _y = _x.pop('pred_target')

            x.append(_x.to_numpy())
            y.append(_y.to_numpy())

        self.cv.cross_validate(x=np.concatenate(x), y=np.concatenate(y))
        self.metrics.clear()
        self.metrics.update(self.cv.metrics.metrics)

    def dump_result(self, market_date: datetime.date):
        pass

    def reset(self):
        """
        Resets the factor and factor_value data.
        """
        self.factor.clear()
        MDS.clear()
        self.cv.clear()

    def run(self):
        """
        Runs the factor validation process.
        """
        # self.initialize_factor()

        calendar = simulated_env.trade_calendar(start_date=self.start_date, end_date=self.end_date)

        replay = ProgressiveReplay(
            loader=historical.loader,
            tickers=[],
            dtype=self.dtype.split(','),
            start_date=self.start_date,
            end_date=self.end_date,
            calendar=calendar,
            bod=self.bod,
            eod=self.eod,
            tick_size=0.001,
        )

        for market_data in replay:  # type: MarketData
            if not is_market_session(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)

            idx = int(market_data.timestamp // self.sampling_interval)

            if self.last_idx != idx:
                self._collect_factor()

    def bod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} bod process...')

        # Startup task 0: Update subscription
        self._update_index_weights(market_date=market_date)

        # Backtest specific action 1: Unzip data
        historical.extract_archive(market_date=market_date, stock_pool=self.index_weights.keys(), dtype='All')

        # Startup task 2: Update subscription and replay
        self._update_subscription(replay=replay)

        # Startup task 3: Update caches
        self.initialize_factor()

        # start the manager
        MDS.monitor_manager.start()

    def eod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:
        random.seed(42)
        LOGGER.info(f'Starting {market_date} eod process...')

        # stop the manager
        MDS.monitor_manager.stop()

        self.validation(market_date=market_date)

        self.dump_result(market_date=market_date)

        self.reset()


def main():
    """
    Main function to run factor validation or batch validation.
    """
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 2, 1)

    from .TradeFlow.momentum import MomentumSupportMonitor

    factor = MomentumSupportMonitor(sampling_interval=60, sample_size=50)

    validator = FactorValidation(start_date=start_date, end_date=end_date, factor=factor)

    # validator = FactorValidation(
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    # validator = FactorBatchValidation(
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    # validator = InterTemporalValidation(
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 4, 1),
    #     training_days=5,
    # )

    # validator = FactorValidatorExperiment(
    #     override_cache=True,
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    # validator.factor.append(some_new_factor)

    validator.run()
    safe_exit()


if __name__ == '__main__':
    main()

def collect_factor(monitors: dict[str, MarketDataMonitor] | list[MarketDataMonitor] | MarketDataMonitor) -> dict[str, float]:
    factors = {}

    if isinstance(monitors, dict):
        monitors = list(monitors.values())
    elif isinstance(monitors, FactorMonitor):
        monitors = [monitors]

    for monitor in monitors:
        if monitor.is_ready and monitor.enabled:
            if DEBUG_MODE and monitor.serializable and not monitor.is_sync:
                monitor.memory_core.from_shm()

            factor_value = monitor.value
            name = monitor.name.removeprefix('Monitor.')

            if isinstance(factor_value, (int, float)):
                factors[name] = factor_value
            elif isinstance(factor_value, dict):
                # FactorPoolDummyMonitor having hard coded name
                if monitor.name == 'Monitor.FactorPool.Dummy':
                    factors.update(factor_value)
                # synthetic index monitor should have duplicated logs
                elif monitor.__class__.__name__ == 'SyntheticIndexMonitor':
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
                    factors.update({f'{monitor.index_name}.{key}': value for key, value in factor_value.items()})
                else:
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
            else:
                raise NotImplementedError(f'Invalid return type, expect float | dict[str, float], got {type(factor_value)}.')

    return factors

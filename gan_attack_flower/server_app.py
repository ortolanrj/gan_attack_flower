"""Flower ServerApp: FedAvg entre uma vítima e um atacante."""

from __future__ import annotations

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .task import CNNClassifier, get_weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    out: Metrics = {}
    keys = {k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))}
    for k in keys:
        out[k] = sum(
            n * float(m[k]) for n, m in metrics if k in m and isinstance(m[k], (int, float))
        ) / total
    return out


def server_fn(context: Context) -> ServerAppComponents:
    rc = context.run_config
    num_rounds = int(rc["num-server-rounds"])

    initial_parameters = ndarrays_to_parameters(get_weights(CNNClassifier()))
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)

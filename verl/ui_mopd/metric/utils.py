"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from typing import Any

import numpy as np


def process_validation_metrics_global(data_sources: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "acc": accuracy

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics_with_acc(data_sources, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2var2vals = defaultdict(lambda: defaultdict(list))
    for sample_idx, data_source in enumerate(data_sources):
        var2vals = data_src2var2vals[data_source]
        for var_name, var_vals in infos_dict.items():
            if var_name in ["acc", "score"]:
                var2vals[var_name].append(var_vals[sample_idx])

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2vals in data_src2var2vals.items():
        for var_name, vals in var2vals.items():
            data_src2var2metric2val[data_source][var_name]['mean'] = np.mean(vals)
            data_src2var2metric2val[data_source][var_name]['std'] = np.std(vals)

    return data_src2var2metric2val

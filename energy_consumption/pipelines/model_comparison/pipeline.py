"""
This module defines the model comparison pipeline.
"""
from kedro.pipeline import Pipeline, node
from .nodes import compare_models


def create_pipeline(**kwargs):
    """
    Create the model comparison pipeline.
    
    Returns:
        A Pipeline object
    """
    return Pipeline(
        [
            node(
                func=compare_models,
                inputs=["arima_results", "sarima_results", "prophet_results", "xgboost_results", "rnn_results"],
                outputs="comparison_results",
                name="compare_models_node",
            ),
        ]
    )
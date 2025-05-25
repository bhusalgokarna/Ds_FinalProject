"""
This module defines the model training pipeline.
"""
from kedro.pipeline import Pipeline, node
from .nodes import train_arima_models, train_sarima_models, train_prophet_models, train_xgboost_models, train_rnn_models


def create_pipeline(**kwargs):
    """
    Create the model training pipeline.
    
    Returns:
        A Pipeline object
    """
    return Pipeline(
        [
            node(
                func=train_arima_models,
                inputs=["arima_data", "params:arima"],
                outputs="arima_results",
                name="train_arima_models_node",
            ),
            node(
                func=train_sarima_models,
                inputs=["arima_data", "params:sarima"],
                outputs="sarima_results",
                name="train_sarima_models_node",
            ),
            node(
                func=train_prophet_models,
                inputs=["prophet_data", "params:prophet"],
                outputs="prophet_results",
                name="train_prophet_models_node",
            ),
            node(
                func=train_xgboost_models,
                inputs=["ml_data", "params:xgboost"],
                outputs="xgboost_results",
                name="train_xgboost_models_node",
            ),
            node(
                func=train_rnn_models,
                inputs=["rnn_data", "params:rnn"],
                outputs="rnn_results",
                name="train_rnn_models_node",
            ),
        ]
    )
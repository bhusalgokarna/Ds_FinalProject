"""
This module defines the feature engineering pipeline.
"""
from kedro.pipeline import Pipeline, node
from .nodes import prepare_arima_data, prepare_ml_data, create_sequences, prepare_prophet_data, create_sequences_with_temperature


def create_pipeline(**kwargs):
    """
    Create the feature engineering pipeline.
    
    Returns:
        A Pipeline object
    """
    return Pipeline(
        [
            node(
                func=prepare_arima_data,
                inputs="split_data",
                outputs="arima_data",
                name="prepare_arima_data_node",
            ),
            node(
                func=prepare_ml_data,
                inputs=["split_data", "feature_data", "params:lag_features"],
                outputs="ml_data",
                name="prepare_ml_data_node",
            ),
            node(
                func=create_sequences_with_temperature,
                inputs=["split_data", "feature_data", "params:sequence_length"],
                outputs="rnn_data",
                name="create_sequences_with_temperature_node",
            ),
            node(
                func=prepare_prophet_data,
                inputs="split_data",
                outputs="prophet_data",
                name="prepare_prophet_data_node",
            ),
        ]
    )
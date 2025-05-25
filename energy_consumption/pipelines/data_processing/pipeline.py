"""
This module defines the data processing pipeline.
"""
from kedro.pipeline import Pipeline, node
from .nodes import download_data, preprocess_data, add_time_features, create_resampled_datasets, split_data, download_temperature_data, preprocess_temperature_data, merge_energy_temperature_data


def create_pipeline(**kwargs):
    """
    Create the data processing pipeline.
    
    Returns:
        A Pipeline object
    """
    return Pipeline(
        [
            node(
                func=download_data,
                inputs="params:data_url",
                outputs="raw_data",
                name="download_data_node",
            ),
            node(
                func=preprocess_data,
                inputs="raw_data",
                outputs="preprocessed_energy_data",
                name="preprocess_data_node",
            ),
            node(
                func=download_temperature_data,
                inputs="params:temperature_data_url",
                outputs="raw_temperature_data",
                name="download_temperature_data_node",
            ),
            node(
                func=preprocess_temperature_data,
                inputs="raw_temperature_data",
                outputs="preprocessed_temperature_data",
                name="preprocess_temperature_data_node",
            ),
            node(
                func=merge_energy_temperature_data,
                inputs=["preprocessed_energy_data", "preprocessed_temperature_data"],
                outputs="merged_data",
                name="merge_energy_temperature_data_node",
            ),
            node(
                func=add_time_features,
                inputs="merged_data",
                outputs="feature_data",
                name="add_time_features_node",
            ),
            node(
                func=create_resampled_datasets,
                inputs=["feature_data", "params:target_column"],
                outputs="resampled_data",
                name="create_resampled_datasets_node",
            ),
            node(
                func=split_data,
                inputs=["resampled_data", "params:test_size"],
                outputs="split_data",
                name="split_data_node",
            ),
        ]
    )
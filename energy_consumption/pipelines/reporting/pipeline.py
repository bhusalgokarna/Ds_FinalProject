"""
This module defines the reporting pipeline.
"""
from kedro.pipeline import Pipeline, node
from .nodes import create_report, analyze_temperature_correlation, plot_model_comparison


def create_pipeline(**kwargs):
    """
    Create the reporting pipeline.
    
    Returns:
        A Pipeline object
    """
    return Pipeline(
        [
            node(
                func=analyze_temperature_correlation,
                inputs="feature_data",
                outputs=None,
                name="analyze_temperature_correlation_node",
            ),
            node(
                func=create_report,
                inputs=[
                    "comparison_results",
                    "arima_results",
                    "sarima_results",
                    "prophet_results",
                    "xgboost_results",
                    "rnn_results"
                ],
                outputs="report",
                name="create_report_node",
            ),
        ]
    )
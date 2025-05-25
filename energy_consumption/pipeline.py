"""
This module defines the main pipeline.
"""
from kedro.pipeline import Pipeline

from energy_consumption.pipelines import data_processing as dp
from energy_consumption.pipelines import feature_engineering as fe
from energy_consumption.pipelines import model_training as mt
from energy_consumption.pipelines import model_comparison as mc
from energy_consumption.pipelines import reporting as rp


def create_pipeline(**kwargs):
    """
    Create the main pipeline.
    
    Returns:
        A Pipeline object
    """
    data_processing_pipeline = dp.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_comparison_pipeline = mc.create_pipeline()
    reporting_pipeline = rp.create_pipeline()
    
    return Pipeline(
        [
            data_processing_pipeline,
            feature_engineering_pipeline,
            model_training_pipeline,
            model_comparison_pipeline,
            reporting_pipeline,
        ]
    )
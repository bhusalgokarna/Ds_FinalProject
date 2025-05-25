"""
This module contains the model comparison nodes for the energy consumption pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_models(arima_results: Dict[str, Any],
                  sarima_results: Dict[str, Any],
                  prophet_results: Dict[str, Any],
                  xgboost_results: Dict[str, Any],
                  rnn_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the performance of different models.
    
    Args:
        arima_results: Dictionary containing ARIMA model results
        sarima_results: Dictionary containing SARIMA model results
        prophet_results: Dictionary containing Prophet model results
        xgboost_results: Dictionary containing XGBoost model results
        rnn_results: Dictionary containing RNN model results
        
    Returns:
        Dictionary containing model comparison results
    """
    logger.info("Comparing model performance")
    
    # List of model types
    model_types = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "RNN"]
    
    # Dictionary to store results
    comparison_results = {}
    
    # List of frequencies
    frequencies = list(arima_results.keys())
    
    for freq in frequencies:
        logger.info(f"Comparing models for {freq}")
        
        # Get results for each model
        arima_result = arima_results[freq]
        sarima_result = sarima_results[freq]
        prophet_result = prophet_results[freq]
        xgboost_result = xgboost_results[freq]
        rnn_result = rnn_results[freq]
        
        # Create a DataFrame to compare metrics
        metrics_df = pd.DataFrame(index=model_types)
        
        # Add metrics for each model
        metrics_df.loc["ARIMA", "MSE"] = arima_result["metrics"]["mse"]
        metrics_df.loc["ARIMA", "RMSE"] = arima_result["metrics"]["rmse"]
        metrics_df.loc["ARIMA", "MAE"] = arima_result["metrics"]["mae"]
        metrics_df.loc["ARIMA", "R²"] = arima_result["metrics"]["r2"]
        
        metrics_df.loc["SARIMA", "MSE"] = sarima_result["metrics"]["mse"]
        metrics_df.loc["SARIMA", "RMSE"] = sarima_result["metrics"]["rmse"]
        metrics_df.loc["SARIMA", "MAE"] = sarima_result["metrics"]["mae"]
        metrics_df.loc["SARIMA", "R²"] = sarima_result["metrics"]["r2"]
        
        metrics_df.loc["Prophet", "MSE"] = prophet_result["metrics"]["mse"]
        metrics_df.loc["Prophet", "RMSE"] = prophet_result["metrics"]["rmse"]
        metrics_df.loc["Prophet", "MAE"] = prophet_result["metrics"]["mae"]
        metrics_df.loc["Prophet", "R²"] = prophet_result["metrics"]["r2"]
        
        metrics_df.loc["XGBoost", "MSE"] = xgboost_result["metrics"]["mse"]
        metrics_df.loc["XGBoost", "RMSE"] = xgboost_result["metrics"]["rmse"]
        metrics_df.loc["XGBoost", "MAE"] = xgboost_result["metrics"]["mae"]
        metrics_df.loc["XGBoost", "R²"] = xgboost_result["metrics"]["r2"]
        
        metrics_df.loc["RNN", "MSE"] = rnn_result["metrics"]["mse"]
        metrics_df.loc["RNN", "RMSE"] = rnn_result["metrics"]["rmse"]
        metrics_df.loc["RNN", "MAE"] = rnn_result["metrics"]["mae"]
        metrics_df.loc["RNN", "R²"] = rnn_result["metrics"]["r2"]
        
        # Find the best model for each metric
        best_mse_model = metrics_df["MSE"].idxmin()
        best_rmse_model = metrics_df["RMSE"].idxmin()
        best_mae_model = metrics_df["MAE"].idxmin()
        best_r2_model = metrics_df["R²"].idxmax()
        
        # Determine the overall best model based on average rank across metrics
        metrics_df["MSE_rank"] = metrics_df["MSE"].rank()
        metrics_df["RMSE_rank"] = metrics_df["RMSE"].rank()
        metrics_df["MAE_rank"] = metrics_df["MAE"].rank()
        metrics_df["R²_rank"] = metrics_df["R²"].rank(ascending=False)
        
        metrics_df["Average_rank"] = metrics_df[["MSE_rank", "RMSE_rank", "MAE_rank", "R²_rank"]].mean(axis=1)
        best_overall_model = metrics_df["Average_rank"].idxmin()
        
        logger.info(f"Best model for {freq} based on average rank: {best_overall_model}")
        
        # Store results
        comparison_results[freq] = {
            "metrics_df": metrics_df,
            "best_mse_model": best_mse_model,
            "best_rmse_model": best_rmse_model,
            "best_mae_model": best_mae_model,
            "best_r2_model": best_r2_model,
            "best_overall_model": best_overall_model,
            "predictions": {
                "ARIMA": arima_result["predictions"],
                "SARIMA": sarima_result["predictions"],
                "Prophet": prophet_result["predictions"],
                "XGBoost": xgboost_result["predictions"],
                "RNN": rnn_result["predictions"]
            },
            "actual": arima_result["actual"]  # All models have the same actual values
        }
    
    # Determine the best model across all frequencies
    overall_ranks = pd.DataFrame(index=model_types, columns=frequencies)
    
    for freq in frequencies:
        for model in model_types:
            overall_ranks.loc[model, freq] = comparison_results[freq]["metrics_df"].loc[model, "Average_rank"]
    
    overall_ranks["Average"] = overall_ranks.mean(axis=1)
    best_model_overall = overall_ranks["Average"].idxmin()
    
    logger.info(f"Best model overall: {best_model_overall}")
    
    comparison_results["overall"] = {
        "overall_ranks": overall_ranks,
        "best_model_overall": best_model_overall
    }
    
    return comparison_results
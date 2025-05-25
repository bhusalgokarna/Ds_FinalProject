"""
This module contains the reporting nodes for the energy consumption pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_temperature_correlation(feature_data: pd.DataFrame) -> None:
    """
    Analyze the correlation between temperature and energy consumption.
    
    Args:
        feature_data: DataFrame with energy and temperature data
    """
    logger.info("Analyzing temperature correlation with energy consumption")
    
    # Check if temperature data is available
    if 'temperature' not in feature_data.columns:
        logger.warning("Temperature data not available for correlation analysis")
        return
    
    # Create directory for plots
    os.makedirs("data/08_reporting", exist_ok=True)
    
    # Resample data to different frequencies
    hourly_data = feature_data.resample('H').mean()
    daily_data = feature_data.resample('D').mean()
    weekly_data = feature_data.resample('W').mean()
    
    # Plot correlation for hourly data
    plt.figure(figsize=(12, 8))
    plt.scatter(hourly_data['temperature'], hourly_data['Global_active_power'], alpha=0.5)
    plt.title('Hourly Temperature vs. Energy Consumption')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Global Active Power')
    plt.grid(True)
    plt.savefig("data/08_reporting/hourly_temp_correlation.png")
    plt.close()
    
    # Plot correlation for daily data
    plt.figure(figsize=(12, 8))
    plt.scatter(daily_data['temperature'], daily_data['Global_active_power'], alpha=0.5)
    plt.title('Daily Temperature vs. Energy Consumption')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Global Active Power')
    plt.grid(True)
    plt.savefig("data/08_reporting/daily_temp_correlation.png")
    plt.close()
    
    # Plot correlation for weekly data
    plt.figure(figsize=(12, 8))
    plt.scatter(weekly_data['temperature'], weekly_data['Global_active_power'], alpha=0.5)
    plt.title('Weekly Temperature vs. Energy Consumption')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Global Active Power')
    plt.grid(True)
    plt.savefig("data/08_reporting/weekly_temp_correlation.png")
    plt.close()
    
    # Calculate correlation coefficients
    hourly_corr = hourly_data[['temperature', 'Global_active_power']].corr().iloc[0, 1]
    daily_corr = daily_data[['temperature', 'Global_active_power']].corr().iloc[0, 1]
    weekly_corr = weekly_data[['temperature', 'Global_active_power']].corr().iloc[0, 1]
    
    logger.info(f"Hourly temperature correlation: {hourly_corr:.4f}")
    logger.info(f"Daily temperature correlation: {daily_corr:.4f}")
    logger.info(f"Weekly temperature correlation: {weekly_corr:.4f}")
    
    # If HDD and CDD are available, plot their correlation too
    if 'HDD' in feature_data.columns and 'CDD' in feature_data.columns:
        # Plot HDD correlation
        plt.figure(figsize=(12, 8))
        plt.scatter(daily_data['HDD'], daily_data['Global_active_power'], alpha=0.5)
        plt.title('Daily Heating Degree Days vs. Energy Consumption')
        plt.xlabel('HDD')
        plt.ylabel('Global Active Power')
        plt.grid(True)
        plt.savefig("data/08_reporting/hdd_correlation.png")
        plt.close()
        
        # Plot CDD correlation
        plt.figure(figsize=(12, 8))
        plt.scatter(daily_data['CDD'], daily_data['Global_active_power'], alpha=0.5)
        plt.title('Daily Cooling Degree Days vs. Energy Consumption')
        plt.xlabel('CDD')
        plt.ylabel('Global Active Power')
        plt.grid(True)
        plt.savefig("data/08_reporting/cdd_correlation.png")
        plt.close()
        
        # Calculate correlation coefficients
        hdd_corr = daily_data[['HDD', 'Global_active_power']].corr().iloc[0, 1]
        cdd_corr = daily_data[['CDD', 'Global_active_power']].corr().iloc[0, 1]
        
        logger.info(f"HDD correlation: {hdd_corr:.4f}")
        logger.info(f"CDD correlation: {cdd_corr:.4f}")

def plot_model_comparison(comparison_results: Dict[str, Any]) -> None:
    """
    Plot model comparison results.
    
    Args:
        comparison_results: Dictionary containing model comparison results
    """
    logger.info("Plotting model comparison results")
    
    # Create directory for plots
    os.makedirs("data/08_reporting", exist_ok=True)
    
    # List of frequencies
    frequencies = [freq for freq in comparison_results.keys() if freq != "overall"]
    
    # Plot metrics comparison for each frequency
    for freq in frequencies:
        metrics_df = comparison_results[freq]["metrics_df"]
        
        # Plot metrics comparison
        plt.figure(figsize=(15, 10))
        
        # MSE
        plt.subplot(2, 2, 1)
        sns.barplot(x=metrics_df.index, y=metrics_df["MSE"])
        plt.title(f"MSE Comparison - {freq}")
        plt.ylabel("MSE")
        plt.xticks(rotation=45)
        
        # RMSE
        plt.subplot(2, 2, 2)
        sns.barplot(x=metrics_df.index, y=metrics_df["RMSE"])
        plt.title(f"RMSE Comparison - {freq}")
        plt.ylabel("RMSE")
        plt.xticks(rotation=45)
        
        # MAE
        plt.subplot(2, 2, 3)
        sns.barplot(x=metrics_df.index, y=metrics_df["MAE"])
        plt.title(f"MAE Comparison - {freq}")
        plt.ylabel("MAE")
        plt.xticks(rotation=45)
        
        # R²
        plt.subplot(2, 2, 4)
        sns.barplot(x=metrics_df.index, y=metrics_df["R²"])
        plt.title(f"R² Comparison - {freq}")
        plt.ylabel("R²")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"data/08_reporting/metrics_comparison_{freq}.png")
        plt.close()
        
        # Plot predictions comparison
        plt.figure(figsize=(15, 8))
        
        # Get actual and predictions
        actual = comparison_results[freq]["actual"]
        predictions = comparison_results[freq]["predictions"]
        
        # Plot actual values
        plt.plot(actual.index, actual, label="Actual", linewidth=2, color="black")
        
        # Plot predictions for each model
        for model, pred in predictions.items():
            if pred is not None:
                plt.plot(actual.index, pred, label=f"{model} Predictions", alpha=0.7)
        
        plt.title(f"Predictions Comparison - {freq}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"data/08_reporting/predictions_comparison_{freq}.png")
        plt.close()
    
    # Plot overall ranks
    overall_ranks = comparison_results["overall"]["overall_ranks"]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(overall_ranks, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Model Ranking Across Frequencies (Lower is Better)")
    plt.tight_layout()
    plt.savefig("data/08_reporting/overall_ranks.png")
    plt.close()

def create_report(comparison_results: Dict[str, Any],
                 arima_results: Dict[str, Any],
                 sarima_results: Dict[str, Any],
                 prophet_results: Dict[str, Any],
                 xgboost_results: Dict[str, Any],
                 rnn_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive report.
    
    Args:
        comparison_results: Dictionary containing model comparison results
        arima_results: Dictionary containing ARIMA model results
        sarima_results: Dictionary containing SARIMA model results
        prophet_results: Dictionary containing Prophet model results
        xgboost_results: Dictionary containing XGBoost model results
        rnn_results: Dictionary containing RNN model results
        
    Returns:
        Dictionary containing report data
    """
    logger.info("Creating comprehensive report")
    
    # Create plots
    plot_model_comparison(comparison_results)
    
    # Create report data
    report = {
        "best_model_overall": comparison_results["overall"]["best_model_overall"],
        "frequencies": {}
    }
    
    # List of frequencies
    frequencies = [freq for freq in comparison_results.keys() if freq != "overall"]
    
    for freq in frequencies:
        metrics_df = comparison_results[freq]["metrics_df"]
        
        # Convert DataFrame to dict for JSON serialization
        metrics_dict = {}
        for model in metrics_df.index:
            metrics_dict[model] = {
                "MSE": float(metrics_df.loc[model, "MSE"]),
                "RMSE": float(metrics_df.loc[model, "RMSE"]),
                "MAE": float(metrics_df.loc[model, "MAE"]),
                "R²": float(metrics_df.loc[model, "R²"])
            }
        
        report["frequencies"][freq] = {
            "best_overall_model": comparison_results[freq]["best_overall_model"],
            "best_mse_model": comparison_results[freq]["best_mse_model"],
            "best_rmse_model": comparison_results[freq]["best_rmse_model"],
            "best_mae_model": comparison_results[freq]["best_mae_model"],
            "best_r2_model": comparison_results[freq]["best_r2_model"],
            "metrics": metrics_dict
        }
    
    # Add RNN best hyperparameters
    rnn_best_params = {}
    for freq in frequencies:
        if rnn_results[freq]["model"] is not None and "best_params" in rnn_results[freq]:
            rnn_best_params[freq] = rnn_results[freq]["best_params"]
    
    report["rnn_best_hyperparameters"] = rnn_best_params
    
    # Add XGBoost feature importances
    xgb_feature_importances = {}
    for freq in frequencies:
        if xgboost_results[freq]["model"] is not None and "feature_importances" in xgboost_results[freq]:
            xgb_feature_importances[freq] = {
                k: float(v) for k, v in xgboost_results[freq]["feature_importances"].items()
            }
    
    report["xgboost_feature_importances"] = xgb_feature_importances
    
    # Save report as JSON
    os.makedirs("data/08_reporting", exist_ok=True)
    with open("data/08_reporting/report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    logger.info("Report saved to data/08_reporting/report.json")
    
    return report
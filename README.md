



#### Time Series Analysis of Power Consumption Data
In this notebook, I'll perform a comprehensive time series analysis on power consumption data using multiple approaches including ARIMA SARIMAX, RNN(LSTM), and Prophet and XGboost models. I'll also use Optuna for hyperparameter tuning to optimize model performance and I will try to make model using auto-arima approach also to see if its works better then with Optuna?

### In this project i have included:

read_data.py => From data loading to the data preprocessing steps everything is happening here in the different pipeline.
ts_eda.ipynb => This is the notbook file where i did the EDA(Exploratory Data Analysis).
ts_model.ipynb => This is the model where i experimented with different model like(ARIMA,SARIMAX, Propet, LSTM, XGboost)


## Data Preparation and Exploratory Analysis
The dataset contains power consumption measurements along with environmental variables like temperature, humidity, and weather conditions. The main variable of interest is `Global_active_power`, which represents the household's global active power consumption in kilowatts. This is the variabble that I am going to forcast in this project.

### From EDA to the Forecasting models
- The data is time-indexed, making it suitable for time series analysis.
- There are multiple variables that could influence power consumption, Many of power related feature are highly depedent features, so i avoid those feature to not have bias. I use only the past values of the Global active power and other environment related features to forcast.
- Global_active_power is highly left scewd, so i use log transformation to make it well distributed. Now it is better but we can see there Bimodal distributions.


Is the data stationary?
- Needed to make 1 diff to make it stationary using adfuller and kpss method.
    - adf_stationary = adf_result[1] < 0.05
    - kpss_stationary = kpss_result[1] > 0.05

### ACF and PACF function
The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots help identify potential AR and MA terms for our ARIMA-based models.

This function shows that the p=<6 q=<2 and as i already needed to 1st difference to our target d=1

The decomposition plot breaks down the time series into trend, seasonality, and residual components, giving us insights into the underlying patterns in the data.


## Model Implementation and Comparison
I've implemented five different time series models to predict power consumption:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: A classic time series model that combines autoregressive (AR), differencing (I), and moving average (MA) components.
I used different approach with Arima model to experiment and to learn and have the best reasult from it.
- 


2. **SARIMAX (Seasonal ARIMA with eXogenous variables)**: Further extends SARIMA by incorporating external variables like temperature and humidity that might influence power consumption.
3. **Prophet**: Facebook's time series forecasting model that handles seasonality and holidays well, and is robust to missing data and outliers.
4. **RNN (LSTM)**: A deep learning approach using Long Short-Term Memory networks, which can capture complex non-linear patterns in the data.


For each model, I have used Optuna for hyperparameter tuning to find the optimal configuration. This automated approach helps us find the best parameters without manual trial and error.

## In-Depth Analysis and Insights

### Temporal Patterns in Power Consumption

1. **Daily Patterns**: Power consumption shows a clear daily cycle, with peaks in the morning and evening when people are typically at home and active. The lowest consumption occurs during the night when most people are sleeping.
2. **Weekly Patterns**: There's a noticeable difference between weekdays and weekends. Weekends showed higher consumption as people spend more time at home.
3. **Seasonal Patterns**: Power consumption is higher during winter months (December-February) and lower during summer months (June-August). This is likely due to heating requirements during colder months.


### Environmental Factors
1. **Temperature**: There's a negative correlation(-12) between temperature and power consumption, confirming that colder temperatures lead to higher power usage, primarily for heating.
2. **Humidity**: Humidity shows a weaker correlation with power consumption compared to temperature, but still plays a role in overall energy usage patterns. But temperature and humidity themself have also negative strong  correlation.


### Sub-metering Analysis
1. **Distribution**: The water heater and air conditioning system (Sub_metering_3) consume the largest portion of electricity, followed by the laundry room (Sub_metering_2) and kitchen (Sub_metering_1).
2. **Seasonal Variations**: The water heater and AC show the most significant seasonal variations, with higher usage during winter months.


### Anomaly Detection

The IQR method identified several anomalies in power consumption, which could represent:
- Unusually high consumption during special events or extreme weather
- Unusually low consumption during vacations or power outages
- Potential measurement errors or equipment malfunctions


### Feature Importance
The Random Forest analysis revealed that the most important features for predicting power consumption are:
1. Temperature
2. Month (seasonality)
3. Day of week
4. Seasion
5. other lags value....
6. Holidays 





## Conclusion

This comprehensive time series analysis of power consumption data has revealed significant patterns and relationships that can be leveraged for accurate forecasting and energy optimization. The comparison of multiple modeling approaches showed that while deep learning models like LSTM can achieve the highest accuracy, simpler models like Prophet offer a good balance of performance and interpretability.

The analysis confirmed the strong influence of temporal patterns (daily, weekly, seasonal) and environmental factors (especially temperature) on power consumption. These insights can be used to develop more efficient energy management strategies and improve forecasting accuracy.
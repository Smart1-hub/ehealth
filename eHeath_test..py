import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from climate_indices import compute, indices, utils
import xarray as xr
import numpy as np
import cartopy.crs as ccrs

file_path = r"C:\Users\DEMO\Desktop\2025 eHealth"
df = pd.read_excel(file_path, sheet_name='in')

df.head()

df['MERRA-2 Temperature at 2 Meters (C)', 'MERRA-2 Temperature at 2 Meters Maximum (C)',
'MERRA-2 Temperature at 2 Meters Minimum (C)', 'MERRA-2 Precipitation Corrected (mm/day)',
'MERRA-2 Relative Humidity at 2 Meters (%)', 'MERRA-2 Specific Humidity at 2 Meters (g/kg)',
'MERRA-2 Wind Speed at 10 Meters (m/s)'] = df['MERRA-2 Temperature at 2 Meters (C)', 'MERRA-2 Temperature at 2 Meters Maximum (C)',
'MERRA-2 Temperature at 2 Meters Minimum (C)', 'MERRA-2 Precipitation Corrected (mm/day)',
'MERRA-2 Relative Humidity at 2 Meters (%)', 'MERRA-2 Specific Humidity at 2 Meters (g/kg)',
'MERRA-2 Wind Speed at 10 Meters (m/s)'].apply(to_numeric, errors = 'coerce')

# Grouping by year and mean values
yearly_trends = df.groupby('YEAR')[['MERRA-2 Temperature at 2 Meters (C)', 'MERRA-2 Temperature at 2 Meters Maximum (C)',
'MERRA-2 Temperature at 2 Meters Minimum (C)', 'MERRA-2 Precipitation Corrected (mm/day)',
'MERRA-2 Relative Humidity at 2 Meters (%)', 'MERRA-2 Specific Humidity at 2 Meters (g/kg)',
'MERRA-2 Wind Speed at 10 Meters (m/s)']].mean()

# Trends over time
plt.figure(figsize=(12, 8))
for column in yearly_trends.columns[1:]: plt.plot(yearly_trends['YEAR'], yearly_trends[column], label=column)
plt.title("Yearly Climate Trends")
plt.xlabel("Year")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Time series analysis (Combine YEAR, MONTH, DAY into a datetime column)
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df.set_index('DATE', inplace=True)

df = df.sort_index()

# Resample to monthly frequency and take the mean
monthly_df = df.resample('M').mean()

# Plot a few key variables
monthly_df[['MERRA-2 Temperature at 2 Meters (C)',
            'MERRA-2 Precipitation Corrected (mm/day)',
            'MERRA-2 Relative Humidity at 2 Meters (%)']].plot(
    subplots=True, figsize=(12, 10), title='Monthly Climate Variables'
)
plt.tight_layout()
plt.show()

# Seasonal decomposition example (temperature)
result = seasonal_decompose(monthly_df['MERRA-2 Temperature at 2 Meters (C)'], model='additive')
result.plot()
plt.suptitle("Seasonal Decomposition - Temperature", fontsize=16)
plt.tight_layout()
plt.show()

# ARIMA MODELING
# Create datetime index
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df.set_index('DATE', inplace=True)
df = df.sort_index()

# Resample to monthly mean temperature
ts = df['MERRA-2 Temperature at 2 Meters (C)'].resample('M').mean()

# Check stationarity using Augmented Dickey-Fuller test
result = adfuller(ts.dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# If not stationary, apply differencing (optional depending on p-value)
ts_diff = ts.diff().dropna()

# Fit ARIMA model (adjust order as needed)
model = ARIMA(ts, order=(1, 1, 1))  # p=1, d=1, q=1 as a starting point
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)

# Plot original series and forecast
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Observed')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast - Monthly Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Correlation
climate_vars = df[[
    'MERRA-2 Temperature at 2 Meters (C)',
    'MERRA-2 Temperature at 2 Meters Maximum (C)',
    'MERRA-2 Temperature at 2 Meters Minimum (C)',
    'MERRA-2 Temperature at 2 Meters Range (C)',
    'MERRA-2 Precipitation Corrected (mm/day)',
    'MERRA-2 Relative Humidity at 2 Meters (%)',
    'MERRA-2 Specific Humidity at 2 Meters (g/kg)',
    'MERRA-2 Wind Speed at 10 Meters (m/s)'
]]

# Calculate correlation matrix
corr_matrix = climate_vars.corr()

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Climate Variables')
plt.tight_layout()
plt.show()

#CLIMATE INDEX
# Resample to monthly total precipitation
monthly_precip = df['MERRA-2 Precipitation Corrected (mm/day)'].resample('M').sum()

# Fill or mask missing values
precip_values = monthly_precip.values
precip_values[np.isnan(precip_values)] = utils.MISSING

# Parameters
scale = 1  # SPI-1

distribution = indices.Distribution.gamma
data_start_year = monthly_precip.index[0].year
data_start_month = monthly_precip.index[0].month

# Compute SPI
spi_values = compute.spi(precip_values, scale, data_start_year, data_start_month, distribution)

# SPI to time series
spi_series = pd.Series(spi_values, index=monthly_precip.index)

# SPI Plot
plt.figure(figsize=(12, 6))
plt.plot(spi_series, label='SPI-1')
plt.axhline(0, color='black', linestyle='--')
plt.title('Standardized Precipitation Index (SPI-1)')
plt.ylabel('SPI Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#GCM EXTRACTION and ANALYSIS 

# === STEP 1: LOAD DATA ===
nc_path = 'C:\Users\DEMO\Desktop\2025 eHealth\CMIP6\tasmin_Amon_MIROC6_ssp245_r1i1p1f1_gn_20150516-20490516.nc'  
ds = xr.open_dataset(nc_path)

# === STEP 2: EXTRACT REGION ===
lat_range = slice(7.5, 8.5)  
lon_range = slice(6.0, 7.5)
# Subset by lat/lon and time
region = ds.sel(lat=lat_range, lon=lon_range)
future = region.sel(time=slice("2020-01-01", "2049-12-31"))

# === STEP 3: PROCESS VARIABLE ===
if 'tas' in ds.data_vars:
    future_data = future['tas'] - 273.15
    var_name = 'Temperature (°C)'
elif 'pr' in ds.data_vars:
    future_data = future['pr'] * 86400  # kg/m²/s to mm/day
    var_name = 'Precipitation (mm/day)'
else:
    raise ValueError("Expected 'tas' or 'pr' in data variables")

# === STEP 4: AGGREGATE TO ANNUAL MEAN ===
annual_mean = future_data.groupby('time.year').mean(dim='time')
regional_mean = annual_mean.mean(dim=['lat', 'lon'])

# === STEP 5: PLOT TIME SERIES ===
plt.figure(figsize=(10, 5))
regional_mean.plot(marker='o')
plt.title(f"CMIP6 Projected {var_name} (2020–2049)")
plt.xlabel("Year")
plt.ylabel(var_name)
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 6: OPTIONAL - MAP AVERAGE SPATIAL DISTRIBUTION ===
mean_spatial = annual_mean.mean(dim='year')
mean_spatial.plot(
    cmap='coolwarm',
    figsize=(8, 6),
    cbar_kwargs={'label': var_name}
)
plt.title(f"Spatial Mean {var_name} (2020–2049)")
plt.show()










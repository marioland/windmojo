
# Historical Forecast API
his API provides access to archived high-resolution weather model data from the Weather Forecast API. The data is continuously archived and updated daily.
https://open-meteo.com/en/docs/historical-forecast-api
https://open-meteo.com/en/docs
https://github.com/open-meteo/open-meteo


Open-Meteo is excited to announce the launch of our new Historical Forecast API. Building on the foundation of our existing weather forecast offerings, this new API provides access to archived high-resolution weather forecast data, making it an invaluable resource for data scientists, researchers, and developers.

What is the Historical Forecast API?
The Historical Forecast API provides seamless access to a vast repository of high-resolution weather model archives. It includes the same weather models used for forecasts but continuously archives all available data, including essential variables like solar radiation, soil moisture, and atmospheric wind data, which are crucial for agricultural and energy applications.

While it functions similarly to the regular Forecast API, the Historical Forecast API is supported by large servers with large storage capacity to retain data. The availability of data varies depending on the weather model. For example, American models like GFS and HRRR offer public archives, allowing us to provide data from 2021 onwards. The data collection is ongoing, but due to the immense volume, this process will take several months more.

As a result, the Historical Forecast API provides incredibly fast access to long time-series weather data for the most important weather forecast models. In an instant, the graph below can be generated, displaying the last six months of data from all major weather forecast models.



Past 6 months of weather models from all major national weather services
This simplicity makes it easy to train machine learning models on historical data and seamlessly combine them with real-time weather forecasts to generate highly optimized forecasts for any specific use case requiring the utmost accuracy. Previously, you would have needed to download and process hundreds of terabytes of GRIB files, which can be very cumbersome to handle.

How is This Data Generated?
Weather models update every 1, 3, or 6 hours. Each "weather model run" is initialized with data from weather stations, satellites, radar, airplanes, soundings, and buoys.

Given that only a small fraction of the Earth is covered by quality weather stations, weather models simulate physical processes to fill in the gaps. Once the global atmospheric state is assimilated, these models predict the next steps in weather development.

Open-Meteo combines all available weather model updates into a continuous time series by taking the initial hours of each update. For instance, if a weather model updates every 3 hours, the first 3 hours of each update are concatenated into a seamless time series.



Combining the first 3 hours of a weather model with 3-hourly updates
The initial time step, "Hour 0," is directly initialized with measurements, while hours 1 and 2 are generated using weather model simulations. Since weather models are highly accurate for the first few hours, the data remains reliable.

In North America and Europe, weather models offer updates every hour. Because "Hour 0" is based on measurements, the data is nearly identical to actual measurements, ensuring high accuracy.

This process results in high-quality weather data with resolutions up to 1 kilometer and global availability. The only drawback is that the data covers only the most recent years.

Difference from the Historical Weather API Based on Reanalysis Data
Experienced users know that Open-Meteo already provides historical weather data through its "Historical Weather API." For novice users expecting a single, definitive source of weather data, this can be confusing.

The "Historical Weather API" is based on reanalysis weather models, particularly ERA5. It offers data from 1940 onwards with consistent quality throughout the time series, making it ideal for analyzing long-term weather trends and climate change. This API focuses on consistency over pinpoint accuracy, with a spatial resolution ranging from 9 to 25 kilometers.

In contrast, the new "Historical Forecast API" is built by continuously assembling high-resolution weather forecast models. With higher update frequencies and increased resolution, this data can be more accurate than reanalysis models. However, it only covers the past 2-5 years and lacks long-term consistency due to the evolving nature of weather models.

Choosing the Right Dataset: Examples:
For analyzing weather trends or climate change over decades: Use the Historical Weather API, which provides reanalysis data from 1940 onwards.

For higher accuracy over the past few years: Opt for the Historical Forecast API, which offers high-resolution forecasts.

For optimizing weather forecasts using machine learning: Utilize data from the same high-resolution weather models available through the new Historical Forecast API.

As Open-Meteo continues to expand its range of weather datasets, maintaining a clear overview can be challenging. If you need assistance navigating this extensive data, don't hesitate to reach out for support.

How is This Data Stored?
Archiving all weather models is a significant challenge. Ensuring correct data downloads and managing immense storage requirements is crucial. The raw data required for this dataset easily exceeds 100 terabytes. This was achievable because Open-Meteo has been diligently storing as much data as possible over the past few years.

To make the data efficiently accessible to the public, Open-Meteo uses a specialized file format and a custom compression algorithm tailored for weather data. This approach surpasses the capabilities of typical databases, offering lightning-fast access times while reducing data size to one-fifth of the original raw data.

The development of the new Historical Forecast API began in November 2022. Initially, data was stored on a dedicated backup server with large but slow hard drives. Over time, more data was added, and the Open-Meteo file format proved essential for managing this volume. Currently, 33 terabytes of highly compressed weather data are available. While this amount of data might not break records, it is a different story when running a fast weather API that offers free access for non-commercial use.

To make the archives accessible, all data has been moved to large storage servers provided by the German hosting provider Hetzner. By combining large, slow HDDs with an SSD-based cache, popular data can be served quickly. Under ideal conditions, accessing two years of weather data takes less than 10 milliseconds. However, if data needs to be read from slow spinning disks, there might be a wait of one or two seconds.

## rate limits
https://open-meteo.com/en/pricing
Free / Open-Access	API Standard	API Professional	API Enterprise
Commercial use	❌	✅	✅	✅
Minutely Limit	⚠️ 600 calls / min	✅ Unlimited	✅ Unlimited	✅ Unlimited
Hourly Limit	⚠️ 5.000 calls / hour	✅ Unlimited	✅ Unlimited	✅ Unlimited
Daily Limit	⚠️ 10.000 calls / day	✅ Unlimited	✅ Unlimited	✅ Unlimited
Monthly Limit	⚠️ 300.000 calls / month	⚠️ 1M calls / month	⚠️ 5M calls / month	✅ >50M calls / month

also concurrency limits 4

## Note on Sea Level Pressure
Mean sea level pressure (msl_pressure) is the atmospheric pressure adjusted to what it would be at sea level, allowing comparisons of pressure readings across different elevations.

## Hourly Weather Variables
Temperature (2 m)
Relative Humidity (2 m)
Dewpoint (2 m)
Apparent Temperature
Precipitation Probability
Precipitation (rain + showers + snow)
Rain
Showers
Snowfall
Snow Depth
Weather code
Sea Level Pressure
Surface Pressure
Cloud Cover Total
Cloud Cover Low
Cloud Cover Mid
Cloud Cover High
Visibility
Evapotranspiration
Reference Evapotranspiration (ET₀)
Vapour Pressure Deficit
Wind Speed (10 m)
Wind Speed (80 m)
Wind Speed (120 m)
Wind Speed (180 m)
Wind Direction (10 m)
Wind Direction (80 m)
Wind Direction (120 m)
Wind Direction (180 m)
Wind Gusts (10 m)
Temperature (80 m)
Temperature (120 m)
Temperature (180 m)
Soil Temperature (0 cm)
Soil Temperature (6 cm)
Soil Temperature (18 cm)
Soil Temperature (54 cm)
Soil Moisture (0-1 cm)
Soil Moisture (1-3 cm)
Soil Moisture (3-9 cm)
Soil Moisture (9-27 cm)
Soil Moisture (27-81 cm)

## Pressure Level Variables
Note: Altitudes are approximate and in meters above sea level (not above ground). Use geopotential_height to get precise altitudes above sea level.
Temperature
Relative Humidity
Cloud Cover
Wind Speed
Wind Direction
Geopotential Height
Wind Speed

1000 hPa (110 m)
975 hPa (320 m)
950 hPa (500 m)
925 hPa (800 m)
900 hPa (1000 m)
850 hPa (1500 m)
800 hPa (1900 m)
700 hPa (3 km)
600 hPa (4.2 km)
500 hPa (5.6 km)
400 hPa (7.2 km)
300 hPa (9.2 km)
250 hPa (10.4 km)
200 hPa (11.8 km)
150 hPa (13.5 km)
100 hPa (15.8 km)
70 hPa (17.7 km)
50 hPa (19.3 km)
30 hPa (22 km)

# Daily Weather Variables
Weather code
Maximum Temperature (2 m)
Minimum Temperature (2 m)
Maximum Apparent Temperature (2 m)
Minimum Apparent Temperature (2 m)
Sunrise
Sunset
Daylight Duration
Sunshine Duration
UV Index
UV Index Clear Sky
Rain Sum
Showers Sum
Snowfall Sum
Precipitation Sum
Precipitation Hours
Precipitation Probability Max
Maximum Wind Speed (10 m)
Maximum Wind Gusts (10 m)
Dominant Wind Direction (10 m)
Shortwave Radiation Sum
Reference Evapotranspiration (ET₀)

## Rate Limit
Das Rate Limit für die Open-Meteo Historical Forecast API im Free Tier liegt bei 600 API-Aufrufen pro Minute, 5.000 Aufrufen pro Stunde und 10.000 Aufrufen pro Tag. Dieses Limit gilt für nicht-kommerzielle Nutzung und es gibt keine Service-Garantien im Free Tier.
# Five years of meteorological surface data at Oak Ridge Reserve in Tennessee
Data curated by: Morgan Steckler (stecklermr@ornl.gov)

PI: Xiao-Ying Yu (yuxiaoying@ornl.gov)

### Data descriptor pre-print: 
"Access to continuous, quality assessed meteorological data is critical for understanding the climatology and atmospheric dynamics of a region. Research facilities like Oak Ridge National Laboratory (ORNL) rely on such data to assess site-specific climatology, model potential emissions, establish safety baselines, and prepare for emergency scenarios. To meet these needs, on-site towers at ORNL collect meteorological data at 15-minute and hourly intervals. However, data measurements from meteorological towers are affected by sensor sensitivity, degradation, lightning strikes, power fluctuations, glitching, and sensor failures, all of which can affect data quality. To address these challenges, we conducted a comprehensive quality assessment and processing of five years of meteorological data collected from ORNL at 15-minute intervals, including measurements of temperature, pressure, humidity, wind, and solar radiation. The time series of each variable was pre-processed and gap-filled using established meteorological data collection and cleaning techniques, i.e., the time series were subjected to structural standardization, data integrity testing, automated and manual outlier detection, and gap-filling. The data product and highly generalizable processing workflow developed in Python Jupyter notebooks are publicly accessible online. As a key contribution of this study, the evaluated 5-year data will be used to train atmospheric dispersion models that simulate dispersion dynamics across the complex ridge-and-valley topography of the Oak Ridge Reservation in East Tennessee."

---

### Quality control processing steps:
1. Standardization
    - 1.1. Header naming
    - 1.2. Index formatting
    - 1.3. Timestep completion
    - 1.4. Null value assignment
2. Quality Control
    - 2.1. Threshold-based integrity testing
    - 2.2. Automated outlier detection
    - 2.3. Manual outlier detection
3. Gap-filling
    - 3.1. Gap-filling with quality controlled hourly data
    - 3.2. Linear interpolation of gaps <= 3 hours

---

## File Structure
The below .zip files contain versions of the hourly and quarter-hourly data at each step of the quality control process. All files were standardized (headers, indices, timesteps, null values). Na-values are represented as -999 and timestampUTC (the index) is formatted in Python as %Y%m%d%H%M%S and has been converted from US/Eastern to UTC.

### Github repository link to step-by-step process: https://github.com/msteckle/orr_met_data_processing
1. `met_towers_2017-2022_hourly-qc.zip`
Hourly-qc data were already quality controlled; they were filtered to desired variables and resampled to 15-minute time steps.

2. `met_towers_2017-2022_original-qc.zip`
Original-qc data went through basic quality control checks using threshold-based integrity testing (see met_inst_ranges.csv in the supplementary.zip for the specific thresholds used for each meterological variable)

3. `met_towers_2017-2022_manual-outlier-id.zip`
Automated outlier detection using moving average windows did not remove ALL the outlying data, i.e., the process produced many false negatives. So, manual-outlier-id contains the indices (start and end dates) of manually identified outlying data points.

4. `met_towers_2017-2022_final-qc.zip`
Final-qc contains the fully quality-controlled quarter-hourly data, where threshold-based integrity testing, automated moving-window average outlier detection, and manual outlier detection were applied. These (or gapfilled-qc) are probably the data you want to use!

5. `met_towers_2017-2022_gapfilled-qc.zip`
The gapfilled-qc data is the final-qc data but with gaps filled using hourly-qc data; if hourly-qc data were not available, linear interpolation was performed on gaps <= 3 hours long.

6. `met_towers_2017-2022_gapfilled-bool.zip`
Boolean table showing which values were gap-filled between final-qa and gapfilled-qa. The loogic is `if gapfilled-qa != final-qa, then gap-filled=True; Otherwise gap-filled=False`

7. `supplementary.zip`
This file contains supplementary data that was used during the quality control process. This include:
    - met_inst_ranges.csv (meteorological instrument ranges), which provides the min/max accepted range of each data variable. For reasoning behind these ranges, see the pre-print linked at the top of this readme.
    - met_towers_info.csv (meteorological towers information), which contains the coordinates, altitude of towers, and altitudes of sensors on each tower. Sensor height from the base of the tower is also included.
    - metadata.csv contains detailed information on the headers found in hourly-qc, original-qc, manual-outlier-id, final-qc, gapfilled-qc, and gapfilled-bool files. It includes short names, long names, descriptions, units, and data types of each meteorological variable. This table is very important for understanding the data in the files.

---

## Column Headers:
More precise information can be found in metadata.csv. But here's a quick summary:
### Meteorological variable short name structure: `{Variable}{Unit}_{0-padded height}m`
- `TempC`         Temperature in celsius
- `BarPresMb`     Barometric pressure in millibars
- `WSpdMph`       Wind speed in miles per hour
- `PkWSpdMph`     Peak wind speed in miles per hour
- `VSSpdMph`      Vertical wind speed in miles per hour
- `WDir`          Wind direction in degrees from North
- `PrecipIn`      Log of precipitation in inches
- `SolarRadWm2`   Solar radiation in watts per meter-squared
- `AbsHum`        Absolute humidity in grams per meter-cubed
- `RelHum`        Relative humidity in percent
- `Sigma`         Sigma Theta (the standard deviation of horizontal wind direction)
- `SigPhi`        Sigma Phi (the standard deviation of 360 wind direction)

## Towers:
More precise information can be found in met_towers_info.csv. Here's a quick summary anyway:
- `TOWA`  Tower A on ORNL campus
- `TOWB`  Tower B on ORNL campus
- `TOWD`  Tower D on ORNL campus
- `TOWF`  Tower F on ridgetop ORNL campus
- `TOWS`  Tower S on ridgetop Y12 campus
- `TOWY`  Tower Y on Y12 campus
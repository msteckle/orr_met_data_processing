# ORR Met Data Processing: Repository Setup Guide

This guide walks through the process of setting up the `orr_met_data_processing` repository for local use, including cloning the repo, downloading required data, setting up the environment, and running the analysis.

---

## Repository Structure after Setup

```
orr_met_data_processing/
├── data/                                           # Downloaded and extracted data files after running setup_data.sh
│   ├── met_towers_2017-2022_final-qc/              # "Final Quality Controlled" tables produced at processing step 01 (pre-gapfilling)
│   ├── met_towers_2017-2022_gapfilled-bool/        # "Gapfilled boolean" tables indicating which values were gap-filled on step 02
│   ├── met_towers_2017-2022_gapfilled-qc/          # "Gapfilled Quality Controlled" tables produced at processing step 02 (post-gapfilling)
│   ├── met_towers_2017-2022_hourly-qc/             # "Hourly Quality Controlled" tables used to fill gaps on step 02
│   ├── met_towers_2017-2022_manual-outlier-id/     # "Manually Identified Outlier" tables manually created for use on step 01
│   ├── met_towers_2017-2022_original-qc/           # "Original Quality Controlled" tables with minimal quality control outputted at step 01
│   ├── supplementary/                              # Extra tower and variable information
│   ├── source_data_015m/                           # Raw 15-minute data used as input into step 01
│   └── source_data_060m/                           # Raw 60-minute data used as input into step 01
├── eda_results/                                    # Generated after running (optional) step 03 jupyter notebook
├── graphics/                                       # Generated after running (optional) step 03 jupyter notebook
├── notebooks/                                      # Ordered Jupyter notebooks to run
├── scripts/                                        # Ordered Python script versions of notebooks (from `jupyter nbconvert`)
├── setup.sh                                        # Bash script to download & extract data necessary for reproduction
├── download_zenodo_archive.sh                        # Bash script to download all data stored on Zenodo
├── environment.yml                                 # Conda environment definition
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

Using SSH (recommended if you've set up an SSH key):

```bash
git clone git@github.com:msteckle/orr_met_data_processing.git
cd orr_met_data_processing
```

> Don't have SSH set up? Follow [GitHub's SSH setup guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

Alternatively, use HTTPS:

```bash
git clone https://github.com/msteckle/orr_met_data_processing.git
cd orr_met_data_processing
```

### 2. Set Up the Conda Environment

```bash
conda env create -f environment.yml
conda activate orr_met_env
```

Alternatively, if you need to update an existing orr_met_env using the environment.yml:
```bash
conda env update -f environment.yml
```

### 3. Download and Extract Data

Run the provided bash script:

```bash
bash setup.sh
```

This script will:
- Create the `data/` directory
- Download [necessary data ZIP files from Zenodo](https://zenodo.org/uploads/15171289)
- Extract them into appropriate subdirectories

### 4. Run the Processing Scripts

Run the notebooks in numerical order from the `notebooks/` directory. To generate all the data, you only need to run the code blocks in `01_outlier_removal.ipynb` and `02_gap_filling.ipynb`. The remaining scripts are optional.

To open the `.ipynb` files and run the scripts block-by-block, use jupyter lab:
```bash
jupyter lab
```

Open the following files and run them block-by-block to re-create our data:
- `01_outlier_removal.ipynb`
- `02_gap_filling.ipynb`


> Alternatively, you can run the corresponding Python scripts in the `scripts/` directory.

### 5. (Optional) Perform Exploratory Data Analysis (EDA)

> Ensure you have enough space on your machine to store ~3.2GB of 100 to 150 dpi images

If you would like to visualize the data, there are many figures that will be produced by `03_eda.ipynb`:

```
orr_met_data_processing/
├── graphics/
│   ├── correlation_heatmaps/     # Heatmaps visualizing the relationship between variables
│   ├── distributions/            # Bar and whisker distribution plots of each tower/variable per a specified time period
│   │   ├── annual/
│   │   ├── monthly/
│   │   ├── multi-annual/
│   ├── gapfilling/           
│   │   ├── pchip_vs_linear/      # Timeseries of Tower D interpolated by PCHIP and linear methods
│   │   ├── missing_data_plots/   # MSNO graphs of missing data for each tower
│   ├── pca/                      # Principal Component Analysis (PCA) graphs
│   ├── timeseries/               # Many 600-dpi images are stored here; expect ~3.5GB of storage usage to produce these
│   │   ├── specific_day/
│   │   ├── random_day/
│   │   ├── monthly/
│   │   ├── annual/
│   │   ├── multi-annual/
```

---

## Notes

- Make sure to have internet access when running the setup script.
- For reproducibility, any edits to the environment should be saved to `environment.yml` with:
  ```bash
  conda env export --from-history > environment.yml
  ```

---

For any issues, open an issue in the [GitHub repository](https://github.com/msteckle/orr_met_data_processing).
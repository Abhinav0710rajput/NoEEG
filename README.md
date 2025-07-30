# NoEEG: A Comprehensive Time Series Analysis and Forecasting Project

This repository, `NoEEG`, is a collection of research and implementation for time series analysis, with a focus on forecasting and imputation. The project explores state-of-the-art models like PatchTST and diffusion models for predicting Depth of Anaesthesia during surgeries. 

## Features

- **Time Series Forecasting:** Implementations and experiments using the PatchTST model.
- **Data Imputation:** Exploration of diffusion models for imputing missing data in time series.
- **Rich Datasets:** Utilizes standard benchmarks like the ETT dataset and custom datasets related to surgeries.
- **Reproducibility:** Jupyter notebooks to reproduce experiments and model training.
- **Pre-trained Models:** Contains pre-trained models for DLinear, Linear, and PatchTST.

## Project Structure

The repository is organized into several key directories:

-   `NoEEG_Borealis/`: Contains the core research materials, experiments, and models.
    -   `DoA-Zero-EEG/`: A self-contained Python project with its own dependencies.
    -   `doa-zero-eeg-sample_filtered/`: Contains sample data in Parquet format and notebooks for data loading and modeling (e.g., with XGBoost).
    -   `PATCH_TST_FINAL_TRAINED/`: Contains pre-trained patch based transformer models and notebooks for testing.
-   `ETDataset-main/`: Contains the ETT dataset and experiments related to it, particularly using PatchTST.
-   `requirements.txt`: A list of Python dependencies for this project.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.8+ and `pip` installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhinav0710rajput/NoEEG.git
    cd NoEEG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to interact with this project is through the Jupyter notebooks located in various subdirectories.

-   To see how to load the Parquet data, check `NoEEG_Borealis/doa-zero-eeg-sample_filtered/load_data.ipynb`.
-   For an example of an XGBoost model, refer to `NoEEG_Borealis/doa-zero-eeg-sample_filtered/XGBOOST_model.ipynb`.
-   To use the pre-trained models, explore the notebooks in `NoEEG_Borealis/PATCH_TST_FINAL_TRAINED/`.



Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".






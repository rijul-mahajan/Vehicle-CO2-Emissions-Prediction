# Vehicle CO2 Emission Predictor

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The **Vehicle CO2 Emission Predictor** is a machine learning-based web application built with Streamlit to predict vehicle emission categories (Low, Moderate, High) based on specifications such as engine size, fuel type, and transmission. The project uses the **Fuel Consumption Ratings 2023** dataset from Natural Resources Canada to train a Random Forest Classifier. The application provides an interactive interface for users to input vehicle details, visualize predictions, and explore dataset and model insights.

## Features

- **Prediction**: Predict emission categories for vehicles based on user inputs.
- **Visualizations**: Interactive charts for feature importance, model performance, emission distribution, and prediction confidence.
- **Dataset Insights**: Statistics and visualizations of the training dataset.
- **Model Information**: Details on model performance, feature correlations, and emission categories.
- **Environmental Impact**: Visual representation of the environmental impact of predicted emissions.

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/rijul-mahajan/vehicle-co2-emission-predictor.git
   cd vehicle-co2-emission-predictor
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   - Obtain the `Fuel_Consumption_Ratings_2023.csv` from [Natural Resources Canada](https://www.nrcan.gc.ca/energy-efficiency/transportation/21008) or another reliable source.
   - Place it in the `data/` directory.

## Usage

1. Train the model:

   ```bash
   python train_and_save.py
   ```

   This script processes the dataset, trains the Random Forest Classifier, and saves the model to `model/vehicle_emission_model.pkl`.

2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   The application will open in your default web browser.

3. Navigate the app:
   - **Home**: Overview of vehicle emissions and key factors.
   - **Dataset Information**: Explore dataset statistics and visualizations.
   - **Model Information**: View model performance and feature importance.
   - **Make Predictions**: Input vehicle specifications to predict emission categories.

## Project Structure

```
vehicle-co2-emission-predictor/
├── data/
│   └── Fuel_Consumption_Ratings_2023.csv
├── model/
│   └── vehicle_emission_model.pkl
├── app.py
├── main.py
├── train_and_save.py
├── requirements.txt
└── README.md
```

- **app.py**: Streamlit application for the web interface.
- **main.py**: Core data processing and model training logic.
- **train_and_save.py**: Script to load data, train the model, and save it.
- **requirements.txt**: List of Python dependencies.
- **data/**: Directory for the dataset.
- **model/**: Directory for the saved model.

## Dataset

The project uses the **Fuel Consumption Ratings 2023** dataset, which includes:

- Vehicle specifications (Make, Model, Engine Size, etc.)
- Fuel consumption metrics (City, Highway, Combined)
- CO2 emissions (g/km)

Ensure the dataset is placed in the `data/` directory before running the training script.

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Engine Size, Fuel Consumption, Transmission Type, Vehicle Class, etc.
- **Target**: Emission categories (Low, Moderate, High) based on CO2 emissions percentiles.
- **Performance**: Evaluated using accuracy and cross-validation scores.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for details.

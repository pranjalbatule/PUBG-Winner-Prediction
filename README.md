
# 🎮 PUBG Winner Prediction

## Dataset
The raw PUBG dataset is not included in this repo due to size (650MB+).
Download it from Kaggle: https://www.kaggle.com/competitions/pubg-finish-placement-prediction/data
Place the file at `data/raw/pubg.csv` before running the preprocessing or training scripts.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app/app.py`
3. Streamlit will print two URLs in the terminal — a Local URL and a Network URL.
   - Local URL: opens the app on your own machine
   - Network URL: lets other devices on the same Wi-Fi/network open the app too (e.g., to demo it on a phone or another laptop)

## Overview

This project predicts a PUBG player's Win Place Percentage using Machine Learning.

## Dataset

PUBG Finish Placement Prediction

## Algorithms Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

## Best Model

XGBoost Regressor

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly

## Folder Structure

```text
app/
data/
images/
models/
notebook/
reports/
src/
```

## Run

```bash
pip install -r requirements.txt

streamlit run app/app.py
```

## Author

Pranjal Batule
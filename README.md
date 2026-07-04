# Weather Temperature Prediction Using SimpleRNN

## Objective
To design, implement, and evaluate a Recurrent Neural Network (RNN) model using SimpleRNN to forecast the next day's temperature based on past weather data.

---

## Dataset
- **Source:** https://www.kaggle.com/datasets/muthuj7/weather-dataset (Kaggle)
- **Raw Data:** 96,453 hourly rows resampled to ~4,000 daily rows
- **Features Used:** Temperature (C), Humidity, Wind Speed (km/h)
- **Target:** Next day's Temperature (C)

---

## Workflow

### Part A: Data Preprocessing
- Parsed datetime and resampled hourly data to daily averages
- Confirmed zero missing values after resampling
- Chronological train/val/test split: **2,813 / 603 / 603 days (70/15/15%)**
- Applied MinMaxScaler fit on training data only to prevent data leakage
- Created sliding window sequences of 7 days → **(2806, 7, 3)** training shape

### Part B: Model Development
| Layer | Details |
|-------|---------|
| Input | (7, 3) — 7 days × 3 features |
| SimpleRNN | 64 units, ReLU activation |
| Dropout | 0.2 |
| Dense | 1 unit, linear activation |

- **Total Parameters:** 4,417
- **Optimizer:** Adam | **Loss:** MSE | **Metric:** MAE
- **Training:** 50 epochs, batch size 32
- **Final Training Loss:** 0.0026 | **Validation Loss:** 0.0018
- No overfitting observed — both losses decreased and stabilized together

### Part C: Evaluation & Forecasting
| Metric | Score |
|--------|-------|
| MAE | 1.51°C |
| RMSE | 1.97°C |
| R² | 0.9461 |

- Predicted vs actual temperatures overlapped closely across all 596 test days
- **7-Day Forecast:** Temperatures ranging from -0.04°C to -0.49°C, correctly continuing the winter cooling trend

---

## Tools & Libraries
Python, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib

---

## Key Takeaway
A lightweight SimpleRNN with only 4,417 parameters achieved an R² of 0.9461, demonstrating that even simple recurrent architectures can effectively learn seasonal temperature patterns from minimal features.

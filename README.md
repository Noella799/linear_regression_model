# Child Internet-Use Early Warning — Regression Model, API & Mobile App

## Mission and problem
Children with anxiety, depression or trauma are often punished as "difficult" because they lack the words to explain what they feel.
My mission is to give kids those words and help the adults around them recognise distress early.
This project predicts a child's problematic-internet-use score (PCIAT, 0–100) from sleep, screen-time, growth and health measures.
School counsellors and parents can use it as an early-warning signal to offer support instead of punishment.

## Dataset
**Child Mind Institute — Problematic Internet Use** (Healthy Brain Network), published on Kaggle.
It contains real data from 3,960 children and teens aged 5–22, with 82 columns: demographics, physical measures, fitness tests, bioelectric impedance, sleep disturbance (SDS), general functioning (CGAS), daily internet use and the Parent–Child Internet Addiction Test (PCIAT).
I use the 2,736 children who have a PCIAT total score as the continuous target, and 10 input features (numeric and categorical).
Source: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data

## API (Swagger UI)
Public Swagger UI: **https://pciat-regression-api.onrender.com/docs**

- `POST /predict`: takes the 10 input variables (typed and range-checked with Pydantic) and returns the predicted PCIAT score and a risk level.
- `POST /retrain`: upload a CSV with the same columns as `summative/API/training_data.csv`. The model continues training on the new rows with `partial_fit`, and `/predict` uses the updated model immediately. A sample file is at `summative/API/sample_new_data.csv`.

The service runs on Render's free plan and sleeps after 15 minutes without traffic, so the first request can take up to a minute.

## Video demo
YouTube: **ADD_YOUR_YOUTUBE_LINK_HERE**

## Run the mobile app
Requirements: Flutter SDK, Android Studio with an Android emulator (or an Android phone with USB debugging enabled).

1. Clone the repository:
   ```bash
   git clone https://github.com/Noella799/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
   ```
2. Install the dependencies:
   ```bash
   flutter pub get
   ```
3. Start an Android emulator (Android Studio → Device Manager → ▶), or connect an Android phone, and check it is detected:
   ```bash
   flutter devices
   ```
4. Run the app:
   ```bash
   flutter run
   ```
   If the emulator shows a blank white screen, run `flutter run --no-enable-impeller` instead.
5. Fill in the 10 fields and tap **Predict**. The prediction, or a readable error for a missing, wrong-type or out-of-range value, appears in the result box under the button.

Example input: age 12, sex 0, daily internet use 3, height 60, weight 100, sleep disturbance 65, CGAS 60, systolic BP 115, heart rate 80, season Fall → predicted score of about 42.4 (Moderate).

## Repository structure
```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml
    ├── uv.lock
    ├── linear_regression/
    │   ├── multivariate.ipynb
    │   └── data/train.csv
    ├── API/
    │   ├── prediction.py
    │   ├── requirements.txt
    │   ├── best_model.pkl
    │   ├── training_data.csv
    │   └── sample_new_data.csv
    └── FlutterApp/
```

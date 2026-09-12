# AgriForecast Model Card

## Model Purpose

AgriForecast is a Streamlit prototype that estimates crop yield for Kenyan farming scenarios using a scikit-learn regression workflow.

## Intended Use

- Educational demonstrations of feature engineering and crop-yield prediction.
- Portfolio review of Streamlit, SQLite, and scikit-learn integration.
- Scenario exploration for generated Kenyan county/crop examples.

## Not Intended For

- Final planting, lending, insurance, or food-security decisions.
- Replacing advice from agronomists or local agricultural extension officers.
- Production use without validation on real, representative, consented agricultural datasets.

## Training Data

The current workflow generates synthetic agricultural records in code. Features include crop type, county, soil pH, soil moisture, fertilizer usage, temperature, rainfall, and altitude.

Because the data is generated, performance metrics describe behavior on simulated examples and should not be interpreted as real-world accuracy.

## Evaluation

The app computes train/test metrics such as R2, MAE, RMSE, and cross-validation scores during model training. These metrics should be reported together with the generated-data caveat.

## Limitations

- Generated data may not capture real pest pressure, soil variation, market practices, climate shocks, or farm management differences.
- County-level climate defaults are simplified.
- The model does not quantify all agronomic uncertainty.
- Predictions may be unstable if input values fall outside the generated training ranges.

## Responsible Use

Present predictions as educational estimates. Clearly label generated-data assumptions in demos, and validate against real local datasets before using results to guide farmer-facing decisions.

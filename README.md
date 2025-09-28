PUBG Game Winner Prediction

Predict the win probability (winPlacePerc) of a PUBG match using player statistics and match metrics. This project demonstrates data analysis, feature engineering, and machine learning model building to understand what influences winning in PUBG.

Project Objective

Analyze PUBG match data to identify player performance patterns.

Predict match win probability using player and match metrics.

Highlight key features that influence winning.

Dataset

Includes player stats and match info such as:

kills, damageDealt, walkDistance, rideDistance, boosts, heals, killPlace, maxPlace, matchDuration, etc.

Target variable: winPlacePerc

Full dataset is large; only a sample or pre-trained model is provided.

Key Features

totalDistance = walkDistance + rideDistance + swimDistance

healItems = heals + boosts

killRatio = kills / totalDistance

killsPerDistance = kills / totalDistance

headshotRate = headshotKills / kills

Models & Performance
Model	R²	RMSE
Linear Regression	0.844	0.0147
Ridge	0.844	0.1212
Lasso	0.836	0.1243
Random Forest	0.915	0.0896
XGBoost (Baseline)	0.9347	0.0785
XGBoost (Tuned)	0.9326	0.0798

Conclusion: Baseline XGBoost performed best; tuning validated robustness.

Challenges & Solutions

Large Dataset: Used subsampling and n_jobs=-1 for parallel processing.

Outliers: Retained outliers as they represent real player behavior; tree-based models handled them well.

Non-linear Relationships: Captured using Random Forest & XGBoost.

Feature Engineering: Added derived metrics to improve prediction accuracy.

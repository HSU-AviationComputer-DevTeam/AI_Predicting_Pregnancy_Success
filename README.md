![image](https://github.com/user-attachments/assets/22834780-59f1-4943-9c4c-249cbd9db863)

## Project Overview
The project aims to predict pregnancy success using advanced machine learning techniques developed for the LG Aimers Hackathon.

## Final Performance
**Private Score: 0.74159**

## 👥 Team Members
<table>
  <tr align="center">
    <td width="150px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
    </td>
  </tr>
  <tr align="center">
    <td>
      김재원
    </td>
    <td>
      신동우
    </td>
      <td>
      홍성인
    </td>
  </tr>
</table>

## Technology Stack
- Python 3.9.7
- pandas 1.3.5
- numpy 1.21.6
- matplotlib 3.5.1
- seaborn 0.11.2
- scikit-learn 1.0.2
- imbalanced-learn 0.8.1
- catboost 1.0.6
- scipy 1.7.3

## Key Features

### Data Preprocessing
- Class-based missing value handling
- Advanced feature engineering
- Strict data leakage prevention

### Modeling Strategy
- 7-fold cross-validation
- CatBoost algorithm
- Ensemble techniques

### Performance Optimization
- Detailed hyperparameter tuning
- Class imbalance handling
- Feature importance-based selection

## Performance Evaluation
- Evaluated using ROC AUC score
- Multiple submission strategies:
  1. Single best-performing model
  2. Average ensemble of all folds
  3. Weighted ensemble of top 3 folds
  4. Weighted ensemble of top 5 folds

## Solution Highlights
- Advanced machine learning techniques
- Sophisticated feature selection
- Robust cross-validation methodology

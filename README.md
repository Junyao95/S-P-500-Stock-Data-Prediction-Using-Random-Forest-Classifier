# S&P 500 Stock Data Prediction Using Random Forest Classifier

This project demonstrates how to apply a Random Forest Classifier to predict stock price movements on S&P 500 data. The code includes a basic backtesting framework that evaluates the model's predictions over time.

## Project Structure
- random_forest_backtest.py: The Python script that contains the code for training, testing, and backtesting the Random Forest model.
- Requirements: Python packages required to run the project.

## Prerequisites
The following Python libraries are required to run the porject:
- pandas 
- scikit-learn
- matplotlib

You can install these libraries using the following command:
```
bash

pip install pandas scikit-learn matplotlib
```
## Data
The project assumes that the S&P 500 stock data is stored in a Pandas DataFrame named sp500. The data should contain columns for Close, Volume, Open, High, Low, and Target (the actual labels for stock movement prediction).

- The Target column is the dependent variable (the direction of stock price movement, e.g., 0 for no increase and 1 for an increase).
- The other columns (Close, Volume, Open, High, Low) are the features used for prediction.

## Code Overview
### Step 1: Initialize the Random Forest Classifier
The RandomForestClassifier from sklearn.ensemble is initialized with the following parameters:
- n_estimators=100: Number of trees in the forest.
- min_samples_split=100: Minimum number of samples required to split an internal node.
- random_state=1: Ensures reproducibility by controlling the randomness of the algorithm.

```
python

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
```
### Step 2: Train/Test Split
The data is split into a training set (train) and a testing set (test). The training set includes all but the last 100 rows of the dataset, and the testing set contains the last 100 rows.

```
python

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
```
### Step 3: Fit the Model and Make Predictions
The model is trained on the training set and predictions are made on the testing set. The predictors are the columns used to predict the target.
```
python

model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])
```
### Step 4: Calculate Precision Score
After making predictions, the precision score is calculated to evaluate the performance of the model on the test set.
```
python

from sklearn.metrics import precision_score
precision_score(test["Target"], preds)
```
### Step 5: Visualization of Predictions
The actual vs predicted values are visualized using a plot for easy comparison.
```
python

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
```
### Step 6: Backtesting Function
A backtest() function is implemented to simulate how the model performs over time. It splits the data into chunks and tests the model's predictions step by step.
```
python

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)
```
### Step 7: Running the Backtest
The backtest() function is called to evaluate the model's performance on the sp500 dataset.
```
python

predictions = backtest(sp500, model, predictors)
```

### How to Run the Script
1. Clone or download the repository.
2. Ensure the S&P 500 stock data is available in a Pandas DataFrame sp500 with the necessary columns (Close, Volume, Open, High, Low, Target).
3. Install the required Python packages using the following command:
```
bash

pip install pandas scikit-learn matplotlib
```
4. Run the Python script:
```
bash

python random_forest_backtest.py
```
### Expected Output
- Model precision score: Printed in the console after running the precision calculation.
- Combined plot: A plot comparing the actual vs predicted stock movement.
- Backtest results: The concatenated results from backtesting, showing how the model performs over multiple time periods.

### Example Workflow
1. Train the Random Forest model on the S&P 500 data using the provided predictors.
2. Make predictions and calculate the precision score.
3. Backtest the model using historical data.
4. Visualize the results to understand the model’s accuracy and consistency over time.

## License

[MIT](https://choosealicense.com/licenses/mit/)

### Screenshot
![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
## Stock-Predictors-Traders

I've developed a LSTM model that analyzes historical data of any particular stock and provides a prediction of the next price value.

## Table of Contents
- [Detailed Description](#detailed-description)
- [Folders](#folders)
- [Evaluation Data](#evaluation-data)
- [Additional Information](#additional-information)

## Detailed Description
I have create an AI to analyze and predit the following next time period (such as next hour, day, week, etc.) that is set by the user. The goal was to create a model that would reliably predict and gain insight on the future price of the finanical market. 

The AI in particular was developed using neural networks, specifically LSTM, using the Pytorch Library in Python. Some of the features that I've included using the library involved Mean Squared Error as the loss function, and Adam as the optimizer. As for preparing the data to trained the model, I utilized pandas_ta and yfinance libraries in order to access finanical data on a particular stock, ETF, etc. The data involved in training involved Technical indicators (such as RSI, SMAs, EMAs, etc. that are commonly used in the financial industry) and price data. While I was able to build upon my current skills, I was able to learn how to quickly learn new topics and apply them to building an AI model and creating an application to solve such problem.

While this project may not be completely ready for practical use on real currency, it has the potential to become more practical with the goal of generating a positive return. 

## Folders

Model:
- Stock_Predictor_JPN.ipynb: Juypter Notebook containing the main code for creating, training, and evaluating the model
- Stock_Predictor_Func.py: Python File that contains all the functions that is used within the Juypter Notebook

Images:
- LSTM_Learning_1000.gif: Animation of the LSTM model learning over time (1000 epochs)
- LSTM_SPY_Optimized.png: Image containing Model predictions during evaluation of SPY with optimized paramaters
- Stock_Predictor_Performance: Image of Confusion Matrix of up and down trends compared of model and actual values

## Evaluation Data
All the evaluation was performed utilizing Matplotlib, and Scikit-Learn
- RMSE: 9.7

This model is better evaluated by viewing plots under Images in the Folders Tab: [Folders](#folders)

## Additional Information
Note that this model shouldn't be used to make financial based decision and is not recommended to use real stock decisions 

Animation of LSTM Model Learning over training period (SPY):

![LSTM_Learning_1000](https://github.com/anguyen1359/Stock-Predictors-Traders/assets/125108200/63cf23d1-df35-41dc-bb47-ec323fdcde76)

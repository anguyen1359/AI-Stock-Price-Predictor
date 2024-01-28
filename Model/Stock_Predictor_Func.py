#%% 
#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn import metrics
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, ArtistAnimation
from IPython.display import HTML


#%% 
#Data Functions

#Loading stock information into dataframe
def load_data(ticker, start_date, end_date = datetime.now().date(), interval="1d"):
    #Stock Data
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    #Technical Indicators
    MACD_results = ta.macd(df['Close'])
    df['MACD_Line'] = MACD_results['MACD_12_26_9']
    df['Signal_Line'] = MACD_results['MACDs_12_26_9']
    df['MACD_Histogram'] = MACD_results['MACDh_12_26_9']

    df['RSI'] = ta.rsi(df['Close'])

    stoch_results = ta.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch_results['STOCHk_14_3_3']
    df['Stoch_D'] = stoch_results['STOCHd_14_3_3']

    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    
    #df['BollingerB_20'] = ta.bbands(df['Close'], length=20)['BBL_20_2.0']
    
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    return df

#Processing dataframe into more useful information
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df['Returns'] = df['Adj Close'].pct_change()*100
    df.dropna(inplace=True)
    return df

#%%
#LSTM Model Functions

#Creating sequences for LSTM Model
def create_sequences(data, seq_length, predict = 'Adj Close'):
    sequences = [] #List to store input sequences
    targets = [] #List to store corresponding target values

    for i in range(len(data) - seq_length):
        #Extracting a sequence of length 'seq_length' from data
        #Giving model past 7 days of information
        seq = data[i:i+seq_length]

        #Target is the next data point after the sequence
        target = data[i+seq_length:i+seq_length+1][predict]

        # Convert the sequence and target to NumPy arrays and add to the lists
        sequences.append(seq.values)
        targets.append(target.values)
        
    # Convert the lists to PyTorch tensors
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

#Model
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(StockPredictor, self).__init__()

        #LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        #ReLu Activation Function
        self.relu = nn.ReLU()
        
        #Connected Linear Layer
        self.Lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        #out = self.relu(out)  # Apply ReLU activation
        out = self.Lin(out[:, -1, :]) #output from last step
        return out
    
#Training
def train_model(model, train_loader, num_epochs, criterion, optimizer, track_losses=False):
    start = datetime.now()
    tracking_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sequence, target in train_loader:
            optimizer.zero_grad()
            outputs = model(sequence)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            #Tracking Losses
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(train_loader)
        tracking_losses.append(average_loss)

        if (epoch+1) % 10 == 0:
            print(f'Epoch: [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    end = datetime.now()
    training_time = end - start
    print(f'Training Time: {training_time}')

    if track_losses:
        return tracking_losses


def finding_best_parameters(model, train_loader, paramaters, epochs, increments, test, targets, process_data, show_data=True):
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(paramaters, lr)

    average_stats = [0.0, 0.0, 0.0, 0.0, 0.0]
    highest_accuracy = 0.0
    highest_accuracy_loc = 0

    all_data = [] #Individual plots
    
    for i in range(increments):
        print(f'Increment: {i+1}/{increments}, Epoch: {epochs*(i)}-{epochs*(i+1)}')
        loss_of_model = train_model(model, train_loader, epochs, criterion, optimizer, track_losses=True)
        
        if show_data:
            loss_over_time(loss_of_model) 

        model.eval()
        with torch.no_grad():
            test_predictions = model(test)
    
        predictions_df = pd.DataFrame({'Date': process_data.index[-len(test_predictions):], 'Predicted': test_predictions.view(-1).numpy()})
        targets_df = pd.DataFrame({'Date': process_data.index[-len(targets):], 'Actual': targets.view(-1).numpy()})

        predictions_df.set_index('Date', inplace=True)
        targets_df.set_index('Date', inplace=True)

        predictions_df = corrected_Prediction_Dates(predictions_df)

        result_df = pd.merge(predictions_df, targets_df, on='Date', how='outer')
        result_df.reset_index(inplace=True)

        all_data.append({
            'result_df': result_df.copy(),
            'increment': i+1,
            'epochs': epochs*(i*1)
        })

        if show_data:
            plt.figure(figsize=(12, 6))
            plt.plot(result_df['Date'], result_df['Actual'], label='Actual')
            plt.plot(result_df['Date'], result_df['Predicted'], label='Predicted')
            plt.title(f'Stock Price Prediction vs Actual')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()
            plt.close()


        predictions_df = add_percent_change(predictions_df)
        targets_df = add_percent_change(targets_df, value = "Actual")

        if show_data:
            stats = calculate_accuracy(predictions_df, targets_df)

            for j in range(len(stats)):
                average_stats[j] += stats[j]
            
            if stats[0] > highest_accuracy:
                highest_accuracy = stats[0]
                highest_accuracy_loc = i+1

        print("\n")
    
    if show_data:
        for k in range(len(average_stats)):
            average_stats[k] /= increments
        
        print(f'Average Accuracy: {average_stats[0]*100:.5f}%')
        print(f'Average Precision: {average_stats[1]*100:.5f}%')
        print(f'Average Sensitivity: {average_stats[2]:.5f}%')
        print(f'Average Specificity: {average_stats[3]:.5f}%')
        print(f'Average F1_score: {average_stats[4]:.5f}%')
        print("\n")
        print(f'Highest Accuracy: {highest_accuracy*100:.5f}% at Increment: {highest_accuracy_loc} (Epochs: {epochs*highest_accuracy_loc})')

    return all_data

def animated_plot(data, speed=100, save=False, save_path=None):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        result_df = data[frame]['result_df']
        increment = data[frame]['increment']
        epochs = data[frame]['epochs']

        line_actual = plt.plot(result_df['Date'], result_df['Actual'], label='Actual')
        line_predicted = plt.plot(result_df['Date'], result_df['Predicted'], label='Predicted')
        plt.title(f'Stock Price Prediction vs Actual (Increment: {increment}, Epochs: {epochs})')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()

        return [line_actual, line_predicted]

    animation = FuncAnimation(fig, update, frames=len(data), interval=speed)
    
    if save:
        animation.save(save_path, writer='ffmpeg', fps=10)

    return HTML(animation.to_jshtml())

#Evaluation
def loss_over_time(losses):
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epoch')
    plt.legend()
    plt.show()

def corrected_Prediction_Dates(df):
    next_date = df.index[-1] + pd.DateOffset(days=1)
    
    df.loc[next_date] = pd.Series(index=df.columns)
    df = df.shift(1)
    df.dropna(inplace=True)

    return df

def add_percent_change(df, value = "Predicted"):

    df["Change (%)"] = df[value].pct_change() * 100
    df["Up/Down (1/0)"] = df["Change (%)"].apply(lambda x: 1 if x > 0.0 else 0)
    return df

def calculate_accuracy(predictions, actual, value = "Up/Down (1/0)"):
    pred = predictions[value]
    act = actual[value]

    pred = pred.iloc[1:-1]
    act = act.iloc[2:]

    matrix = metrics.confusion_matrix(act, pred)

    matrix_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = matrix, display_labels = ["Down", "Up"]
        )
    
    matrix_display.plot()
    plt.show()

    #How often model is correct
    Accuracy = metrics.accuracy_score(act, pred)
    print(f"Accuracy: {Accuracy*100:.5}%")

    #Of the positives predicted, what percentage is truly positive?
    Precision = metrics.precision_score(act, pred)
    print(f"Precision: {Precision*100:.5}%")

    #Sensitivity (sometimes called Recall) measures how good the model is at predicting positives.
    Sensitivity_recall = metrics.recall_score(act, pred)
    print("Sensitivity:", Sensitivity_recall)

    #How well the model is at prediciting negative results?
    Specificity = metrics.recall_score(act, pred, pos_label=0)
    print("Specificity:", Specificity)

    #F-score is the "harmonic mean" of precision and sensitivity.
    F1_score = metrics.f1_score(act, pred)
    print("F1_score:", F1_score)

    return [Accuracy, Precision, Sensitivity_recall, Specificity, F1_score]


#Storing Model
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print("Save Successful")

def load_model(model_to_update, file_path):
    model_to_update.load_state_dict(torch.load(file_path))
    print(f'Updated {model_to_update}')


# %%

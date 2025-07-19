import yfinance as yf  # Downloads stock data
import numpy as np     # Computation 
import pandas as pd    # Handles DataFrame
import matplotlib.pyplot as plt # Plots Data
from statsmodels.tsa.stattools import coint # Useful for cointegration test
from datetime import datetime, timedelta # For handling dates
import time  # Useful for time management
import statsmodels.api as sm # Useful for statiscal linear regression

"""
The model uses the tickers in sp500 tickers, and loops through to find cointegrated pairs,
better models should use the actual  S&P500 stocks to find more pairs or Top N most liquid
stocks over the entire stock market or any other group of correlated stocks in the market

"""
sp500_tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'BRK.B', 'NVDA', 'TSLA', 'JPM', 'V',
    'JNJ', 'WMT', 'BAC', 'PG', 'MA', 'AXP', 'DIS', 'HD', 'KO', 'IBM',
    'INTC', 'GS', 'BA', 'CAT', 'RTX', 'LMT', 'MMM', 'CVX', 'XOM', 'UNH',
    'CVS', 'NKE', 'PYPL', 'GS', 'SPG', 'HCA', 'CCI', 'REGN', 'CTSH', 'LULU',
    'ETN', 'CMCSA', 'FTNT', 'CTAS', 'CTXS', 'NOC', 'NEM', 'PFE', 'MRK', 'CL',
    'CLX', 'KMB', 'KHC', 'PEP', 'SYY', 'ADM']


#@title PairsFinder Class

class PairsFinder(object):
  # Removes duplicate sets of pairs, this is only useful if there are duplicate pairs of stocks within the ticker universe so it cam be removed
  # if there is a unique selection of stocks 
  @staticmethod
  def remove_duplicate_pairs(pairs):   
    # Convert each pair to a frozen set to make it hashable and order-independent
    unique_pairs = set()
    result = []

    for pair in pairs:
        # Convert the pair to a frozenset to ignore order
        pair_set = frozenset(pair)

        # If we haven't seen this pair before, add it to the result
        if pair_set not in unique_pairs:
            unique_pairs.add(pair_set)
            result.append(pair)

    return result


  def load_and_preprocess_data(self, trading_window: int, pairs: int, start, end):
    
    viable_pairs = []
    df = yf.download(sp500_tickers, start=start, end=end)["Close"]
    df = pd.DataFrame(df).dropna(axis=1) # This function downloads all the stocks in the given list of tickers and removes any columns with NAN values
    train = df.iloc[-2* trading_window: -trading_window, :] # Hyperparameter 1, Checks through - Trading window, Trades through last trading window
    test = df.iloc[-trading_window:, :]
    columns = df.columns.unique()

    # Loop through tickers to find viable pairs
    for i in range(len(columns)):
        if len(viable_pairs) == pairs:  # If we've found the required number of pairs, break
            break

        coint_data1 = train[columns[i]]

        for j in range(i+1, len(columns)):
            coint_data2 = train[columns[j]]

            # Skip pairs where either stock has zero variance
            if np.var(coint_data1) == 0 or np.var(coint_data2) == 0:
                continue

            # Perform cointegration test
            score, p_value, lag = coint(coint_data1, coint_data2)

            # If p-value is below threshold, add the pair to the viable pairs list
            if p_value < 0.05:
                viable_pairs.append([columns[i], columns[j], p_value.item()])

                if len(viable_pairs) == pairs:  # Break if the desired number of pairs is met
                    break
    return train, test, self.remove_duplicate_pairs(viable_pairs)
  

class PairsTrading(object):
  def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, pairs: list, shares: float, deviation: float):

    self.pair = pairs[:2] # First 2 columns are the pairs of tickers
    self.p_value = pairs[2] # Last Column is the p_value
    self.shares =  shares # How many shares of stock to bou and sell
    self.trading_data = test_data 
    self.backlog_data  = train_data
    self.deviation = deviation # HyperParameter 2: The standard deviation condition to trade on


  def spread(self):
    ticker1 = self.pair[0]
    ticker2 = self.pair[1]
    coint_data1 = np.array(self.backlog_data[ticker1].values).reshape(-1,1) # Converts the ticker price into an np.array [[p1, p2, ...]]
    coint_data2 = np.array(self.backlog_data[ticker2].values).reshape(-1,1) # Converts the ticker price into an np.array [[p1, p2, ...]]
    hedge_ratio = sm.OLS(coint_data1,sm.add_constant(coint_data2)).fit().params[0] #Fits the data using Linear regression and outputs the beta coefficient
    spread = coint_data1 - hedge_ratio * coint_data2 # Determines the spread of the data

    # Plot the historic spread of the data
    plt.plot(spread)
    plt.title(f"Historic Spread between {ticker1} and {ticker2}")
    plt.axhline(np.mean(spread), color='red', linestyle='--', label='Mean')
    plt.axhline(np.mean(spread) + self.deviation * np.std(spread), color='green', linestyle='--',
                    label=f'Mean + {self.deviation} Std')
    plt.axhline(np.mean(spread) - self.deviation * np.std(spread), color='green', linestyle='--',
                    label=f'Mean - {self.deviation} Std')
    plt.legend()
    plt.show()
    return spread, hedge_ratio


  def stationary_time_series(self):
      historic_spread, hedge_ratio = self.spread()
      d1 = self.trading_data[self.pair[0]].values
      d2 = self.trading_data[self.pair[1]].values
      spread = d1 - hedge_ratio * d2
      # Calculate statistics and trading signal based on the historic spread to avoid look ahead bias
      # The spread to trade on can only be known from past data
      mean = np.mean(historic_spread)
      std = np.std(historic_spread)

      # Define thresholds
      upper_threshold = mean + self.deviation * std
      lower_threshold = mean - self.deviation * std

      # Generate signals
      long_signal = spread < lower_threshold  # Go long the spread
      short_signal =spread > upper_threshold  # Go short the spread
      exit_signal = [False]
      for i in range(1, len(spread)):
        if spread[i-1] <= mean and spread[i] > mean:
          exit_signal.append(True)
        elif spread[i-1] >= mean and spread[i] < mean:
          exit_signal.append(True)
        else:
          exit_signal.append(False)

      # Initialize variables
      # Plot the  actual spread
      plt.plot(spread)
      plt.title(f"Actual Spread between {self.pair[0]} and {self.pair[1]}")
      plt.axhline(mean, color='red', linestyle='--', label='Mean')
      plt.axhline(mean + self.deviation * std, color='green', linestyle='--',
                    label=f'Mean + {self.deviation} Std')
      plt.axhline(mean - self.deviation * std, color='green', linestyle='--',
                    label=f'Mean - {self.deviation} Std')
      plt.legend()
      plt.show()
      return long_signal, short_signal, exit_signal, hedge_ratio


  def indicators(self):
      long_signal, short_signal, exit_signal, hedge_ratio = self.stationary_time_series()
      positions = np.zeros(len(long_signal))  # 1 for long, -1 for short, -2 for exiting an existing position and  0 for no position
      # Generate positions
      for i in range(1, len(positions)):
          if long_signal[i]:
              positions[i] = 1  # Long the spread
          elif short_signal[i]:
              positions[i] = -1  # Short the spread
          elif exit_signal[i]:
              positions[i] = -2  # Exit position
          else:
              positions[i] = 0  # No position

      data = positions
      indicator = []
      historic_position = -2
      for values in range(len(data)):
          if data[values] == 1 and historic_position != data[values]: # If long indicator and I am not already longing the position, Long
              indicator.append("Long")
              historic_position = 1
          elif data[values] == -1 and historic_position != data[values]: # If Short indicator and I am not already shorting the position, Long
              indicator.append("Short")
              historic_position = -1
          elif data[values] == -2 and historic_position != data[values]: # If Exit signal and I have not already exited, exit
              indicator.append("Exit")
              historic_position = -2
          else:
              indicator.append(0)
      indicator[-1] = "Exit" if historic_position != -2 else 0 #Hyperparameter 3: Exit at the end if I havent already exited, you can choose to hold
      return indicator, hedge_ratio

  # Defines the payout of the strategy including the transaction cost 
  def PNL(self, transaction_cost=0):
    pnl_array = np.zeros(len(self.trading_data)) # Creates the cumulative profit array to plot, initialized to all zeros
    indicator, hedge_ratio = self.indicators() # The indicator and hedge ratio function to know when and how much to buy and sell
    entry_gain = 0 # How much profit is made for long short porfolio construction, + for shorting, - for longing
    num_of_trades = 0 # How many trades were made throughout the time series
    data1 = np.array(self.trading_data[self.pair[0]].values) * self.shares # Scales the np array of the values by the number of shares
    data2 = np.array(self.trading_data[self.pair[1]].values) * self.shares # Scales the np array of the values by the number of shares
    pnl = 0 # Total pnl
    for i in range(len(indicator)):
          if indicator[i] == "Long":
              """
              Understanding Entry Gain for Long:
              Longing the stock means borrowing hedge_ratio * data2[i] dollars of stock and buying data1[i] shares of stock
              The payout function works with the cashflow rather than asset value to make it more intuitive
              Assume stock A  = $50, and stock B  = $30 with hedge ratio 1 for simplicity, if I buy stock A I will spend $50 so my cash
              value is -$50, if I borrow stock B and sell it, my cash value is +$30, so my entry gain would be -$20

              With transaction cost taken into account, assume there is a 1% transaction cost, I will gain $30 - 30c for the cost of 
              shorting stock a which is (1-transaction_cost) * hedge_ratio * data2[i] which is $29.7 profit, and it costs $50 + 50c to long
              stock A so in total -$50.5, The entry gain would in total be $29.7-$50.5 = $-21.2    
              """
              if hedge_ratio > 0:
                entry_gain = (hedge_ratio * data2[i] * (1 - transaction_cost) - data1[i]* (1 + transaction_cost))
              else:
                entry_gain = (hedge_ratio * data2[i] - data1[i]) * (1 + transaction_cost)
              num_of_trades += 1
              historic_position = "Long"
            
          elif indicator[i] == "Short":
              """
              Understanding Entry Gain for Short:
              It is essentially the same for Long but the hedge_ratio * data2[i] and the data1[i] is flipped
              """
              if hedge_ratio > 0:
                entry_gain = (data1[i] * (1- transaction_cost) - hedge_ratio * data2[i] * (1 + transaction_cost))
              else:
                entry_gain = (data1[i] - hedge_ratio * data2[i]) * (1- transaction_cost)
              historic_position = "Short"
              num_of_trades += 1
          if indicator[i] == "Exit" and historic_position == "Long":
              """
              Understanding Exit Gain for Long:
              Exiting a long position is the same as entering a short 
              position so the payout should be entry gain for shorting

              """
              if hedge_ratio > 0:
                exit_gain = (hedge_ratio * data2[i] * (1 - transaction_cost) - data1[i]* (1 + transaction_cost))
              else:
                exit_gain = (data1[i] - hedge_ratio * data2[i]) * (1- transaction_cost)
              pnl += entry_gain + exit_gain
              historic_position = "Exit"
              num_of_trades += 1


          elif indicator[i] == "Exit" and historic_position == "Short":
              """
              Understanding Exit Gain for short:
              Exiting a short position is the same as entering a long 
              position so the payout should be entry gain for long

              """
              if hedge_ratio > 0:
                exit_gain = (hedge_ratio * data2[i] * (1 - transaction_cost) - data1[i]* (1 + transaction_cost))
              else:
                exit_gain = (hedge_ratio * data2[i] - data1[i]) * (1 + transaction_cost)
              pnl += entry_gain + exit_gain
              historic_position = "Exit"
              num_of_trades += 1

          pnl_array[i] = pnl
    return pnl_array, num_of_trades


  def metrics(self, pnl_array, num_of_trades):
    dictionary = {}
    Sp500 = yf.download("^GSPC", start=start, end=end)["Close"].iloc[-len(self.trading_data):, :] # Downloads the from the previous trading days to the end of trading period
    model = sm.OLS(Sp500, sm.add_constant(pnl_array)).fit() # Linear regression of the PNL to the S&P500 to calculate the alpha and Beta of the returns
    alpha = model.params.iloc[0] # Alpha of the strategy, should actually use returns rather than total profit; R_s&p = alpha + B * R_Strategy + Error
    beta = model.params.iloc[1] # Correlation coefficient of the PNL series and S&P500
    p1 = self.trading_data[self.pair[0]].iloc[-1] - self.trading_data[self.pair[0]].iloc[0] # Profit for longing A
    p2 = self.trading_data[self.pair[1]].iloc[-1] - self.trading_data[self.pair[1]].iloc[0] # Profit for longing B
    dictionary["Portfolio Value"] = "$" + str(round(pnl_array[-1],2)) # Value of the portfolio at the end of the trading period
    dictionary["Number of Trades"] = num_of_trades # Number of trades made
    dictionary["Trading Days"] = len(self.trading_data) # Number of trading days of the strategy
    dictionary["Sharpe Ratio"] = np.mean(pnl_array) / np.std(pnl_array) #T he sharpe ratio of the strategy
    dictionary["Max Drawdown"] = (np.max(np.maximum.accumulate(pnl_array)) - np.min(pnl_array)) / np.max(pnl_array) # Maximum drawdown of the porfolio
    dictionary["Alpha"] = round(alpha,2) # Alpha of the strategy, higher the better
    dictionary["Beta"] = round(beta, 2) # Beta of the strategy, the lower the better
    dictionary["Volatility"] = np.std(pnl_array) # The total volatility of the Strategy
    dictionary["Boundary of Deviation"] = self.deviation # The predetermined boundary of deviation
    dictionary["Stocks"] = self.pair[0] + " and " + self.pair[1] # The stocks that were traded
    dictionary["P-Value"] = self.p_value # The statistical significance of the cointegration metrics
    dictionary["Long Value"] = "$" +str(round(p1 + p2,2)) # The Value of simply longing both stocks to compare with portfolio Vlaue
    Dataframe = pd.DataFrame(list(dictionary.items()), columns=["Metrics", ""])
    return Dataframe



start = "2012-01-01" # Choice of the Start Date of the data, make sure the amount of data is atleast 2 * the trading window
end = "2024-01-01" # Choice of end date
pairs = 50 # Number of pairs you can select from
trading_window = 550 # Amount of days to trade through
training_data, test_data, pairs = PairsFinder().load_and_preprocess_data(trading_window, pairs, start, end) #Returns the pairs, train, and test data
print(pairs[:5]) # Prints the first 5 Pairs
PT = PairsTrading(training_data, test_data, pairs[0], 1, 1) # Pairs[0] is the specific pair of stocks to trade through

pnl_array,trades = PT.PNL(transaction_cost=0)
plt.title("PNL of Pairs Trading")
plt.plot(pnl_array)
plt.show()

met = PT.metrics(pnl_array, trades)
print(met)
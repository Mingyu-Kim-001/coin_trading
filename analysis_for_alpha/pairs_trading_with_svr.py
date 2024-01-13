#https://medium.com/@financialnoob/pairs-trading-with-support-vector-machines-6cd27c051451

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from time import sleep
import datetime
from utils import get_binance_klines_data_1m, data_freq_convert
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'LTCUSDT', 'TRXUSDT', 'DOTUSDT', 'BNBUSDT']

# %%
dict_df_klines = {}
for symbol in symbols:
  dict_df_klines[symbol] = get_binance_klines_data_1m(symbol, datetime.date(2022, 1, 1), datetime.date(2023 , 12, 31), is_future=True)
  dict_df_klines[symbol] = data_freq_convert(dict_df_klines[symbol], '30T')
timestamp = dict_df_klines[symbols[0]]['timestamp']
print("Data loaded")
# %%
data = pd.DataFrame()
for symbol in symbols:
  data[symbol] = dict_df_klines[symbol]['close']

# %%
data = (data.pct_change()+1).cumprod()
data = data.iloc[1:]
data = data / data.iloc[0]

# %%
# input data
predict_symbol = 'BTCUSDT'
s = [col for col in data.columns if col not in [predict_symbol]]
# target asset
y = [predict_symbol]

# %%
from sklearn.decomposition import PCA

Xtmp = data[s] # select data without the target asset
pca = PCA(n_components=10)
pca.fit(Xtmp)

n_comp = np.arange(1,11)
plt.plot(n_comp, pca.explained_variance_ratio_)
# %%
pca.explained_variance_ratio_

# %%
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from itertools import product

def compute_votes(data, pca_comp, beta, lookback, Cs, gammas, epsilons):
  '''
  compute daily votes of the models with the given parameters
  '''
  # start with equal weights
  weights = np.ones(len(Cs)*len(gammas)*len(epsilons))
  weights = weights/sum(weights) # normalize so that weights sum to 1

  daily_votes = np.zeros(len(data.index))

  for t in tqdm(range(lookback,len(data.index)-1)):
  # for t in range(lookback,len(data.index)-1):
    predictions = []
    for C,gamma,epsilon in product(Cs,gammas,epsilons):
      # print("t", t, "C", C, "gamma", gamma, "epsilon", epsilon)
      model = make_pipeline(StandardScaler(), PCA(n_components=pca_comp), 
                  SVR(C=C, gamma=gamma, epsilon=epsilon))
      X_train = data[s].iloc[t-lookback:t+1].values
      y_train = data[y].iloc[t-lookback:t+1].values.flatten()
      model.fit(X_train,y_train)
      X_test = data[s].iloc[t].values.reshape(1,-1)
      yhat = model.predict(X_test)
      predictions.append(yhat)
    # log all votes
    votes = -np.sign(data[y].iloc[t].values.flatten() - np.array(predictions)) # if price>fair, go short
    final_vote = np.dot(weights,votes)
    daily_votes[t] = final_vote   

    # update weights based on true direction
    true_direction = np.sign((data[y].iloc[t+1] - data[y].iloc[t]).values.flatten()) 
    if final_vote!=true_direction:
      incorrect_votes_ind = np.where(votes!=true_direction)[0]
      weights[incorrect_votes_ind] = beta * weights[incorrect_votes_ind]
      weights = weights/sum(weights)

  return daily_votes

# %%
# SVR hyperparameters
Cs = set((0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000))
gammas = set((0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000))
epsilons = set((0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1))
# strategy parameters
betas = [0.1,0.3,0.5,0.7] # betas for downgrading weights
lookbacks = [20, 30, 40, 50] # how many last trading days to include in model training
pca_comps = [1, 2] # number of principal components to use

# %%
from joblib import Parallel, delayed
import multiprocessing 

def store_results(pca_comp,beta,lookback, lock):
  # results = pd.DataFrame(columns=columns)
  print("pca_comp", pca_comp, "beta", beta, "lookback", lookback)
  daily_votes = compute_votes(data, pca_comp=pca_comp, beta=beta, lookback=lookback, 
                Cs=Cs, gammas=gammas, epsilons=epsilons)
  
  datatmp = data[[predict_symbol]].iloc[lookback+50:].copy() # skip first 50 days
  datatmp['vote'] = daily_votes[lookback+50:]
  datatmp['vote'] = datatmp['vote'].shift()
  datatmp['target_returns'] = datatmp[predict_symbol].pct_change()
  datatmp['alg_returns'] = (np.sign(datatmp['vote']) * datatmp['target_returns'])
  datatmp['alg_cumret'] = np.cumprod(datatmp['alg_returns']+1)
  
  datatmp.dropna(inplace=True)
  
  num_wins = (np.sign(datatmp[['target_returns']].values) == np.sign(datatmp[['vote']].values)).sum()
  num_losses = (np.sign(datatmp[['target_returns']].values) != np.sign(datatmp[['vote']].values)).sum()
  pct_win = num_wins / (num_wins + num_losses)
  avg_win = abs(datatmp[np.sign(datatmp['target_returns']) == np.sign(datatmp['vote'])]['target_returns']).sum()/num_wins
  avg_loss = abs(datatmp[np.sign(datatmp['target_returns']) != np.sign(datatmp['vote'])]['target_returns']).sum()/num_losses
  total_return = (datatmp['alg_cumret'].iloc[-1] - datatmp['alg_cumret'].iloc[0]) / datatmp['alg_cumret'].iloc[0]
  apr = (1+total_return)**(252/len(datatmp.index)) - 1
  sharpe = np.sqrt(252)*datatmp['alg_returns'].mean() / datatmp['alg_returns'].std()
  corrcoef = np.corrcoef(datatmp['target_returns'], datatmp['alg_returns'])[0,1]
  
  results = pd.DataFrame({'Beta':beta, 'Lookback':lookback, 'PCA components':pca_comp, 
                'Num wins':num_wins, 'Num losses':num_losses, 'Pct Win':pct_win, 
                'Avg Win':avg_win, 'Avg Loss':avg_loss, 'Total Return':total_return, 
                'APR':apr, 'Sharpe':sharpe, 'Correlation with traded asset':corrcoef}, index=[0])
  # put results in csv file
  datatmp['timestamp'] = timestamp[lookback+50:]
  datatmp['beta'] = beta
  datatmp['lookback'] = lookback
  datatmp['pca_comp'] = pca_comp
  with lock:
    datatmp.to_csv('data.csv', mode='a', header=False)
    results.to_csv('results.csv', mode='a', header=False)



# %%
param_list = list(product(pca_comps, betas, lookbacks))
param_exclude_list = [(0.1, 20, 1), (0.3,20,1), (0.3,40,1), (0.1,40,1), (0.1,30,1), (0.3,30,1)]
params_list = [param for param in param_list if param not in param_exclude_list]
def run_multiprocessing_tasks(processes):
    # Use a Manager to create a shared Lock
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        # Set up a pool of workers
        with multiprocessing.Pool(processes) as pool:
            # Map the store_results function to the parameter combinations
            # Pass the lock as one of the arguments to each call
            tasks = [(pca_comp, beta, lookback, lock) for pca_comp, beta, lookback in params_list]
            pool.starmap(store_results, tasks)

# This check is crucial for multiprocessing on macOS and Windows
if __name__ == '__main__':
    run_multiprocessing_tasks(6)
# %%

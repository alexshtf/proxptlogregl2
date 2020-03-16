import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import adagrad_experiment
import dataset
import prox_logit_experiment

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ds, dim = dataset.adult_income()

# setup experiment parameters
epochs = range(0, 20)
attempts = range(0, 20)
lrs = np.geomspace(0.001, 10, num=30)
reg_coef = 1

losses = adagrad_experiment.run(dim, ds, epochs, attempts, lrs, reg_coef)
losses.to_csv('adagrad.csv')

losses = prox_logit_experiment.run(dim.shape[1], ds, epochs, attempts, lrs, reg_coef)
losses.to_csv('prox_logit.csv')

# we save results to CSV files, so that we can later visualize them
# without re-running the algorithms, which may take days to run.

prox_losses = pd.read_csv('prox_logit.csv')
best_prox_loss = prox_losses[['lr', 'attempt', 'loss']]\
    .groupby(['lr', 'attempt'], as_index=False)\
    .min()
best_prox_loss['Algorithm'] = 'SPP'

adagrad_losses = pd.read_csv('adagrad.csv')
best_adagrad_loss = adagrad_losses[['lr', 'attempt', 'loss']].groupby(['lr', 'attempt'], as_index=False).min()
best_adagrad_loss['Algorithm'] = 'Adagrad'

best_loss_df = pd.concat([best_prox_loss, best_adagrad_loss], axis=0)

sns.set()
ax = sns.lineplot(x='lr', y='loss', hue='Algorithm', data=best_loss_df, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

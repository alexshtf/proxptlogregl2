import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adagrad


def run(dim, ds, epochs, attempts, lrs, reg_coef):
    losses = pd.DataFrame(columns=['lr', 'epoch', 'attempt', 'loss'])
    total_epochs = len(lrs) * len(attempts) * len(epochs)
    with tqdm(total=total_epochs, desc='lr = NA, attempt = NA, epoch = NA, loss = NA', unit='epochs',
              ncols=140) as pbar:
        for lr in lrs:
            for attempt in attempts:
                x = torch.empty(dim, requires_grad=True, dtype=torch.double)
                torch.nn.init.normal_(x)
                opt = Adagrad([x], lr=lr)

                for epoch in epochs:
                    train_loss = 0
                    for X, y in DataLoader(ds, shuffle=True, batch_size=1):
                        XX = X.squeeze(0)
                        if y.item() == 0:
                            a = -XX
                        else:
                            a = XX

                        score = torch.dot(a, x)
                        loss = torch.log1p(torch.exp(score)) + (reg_coef / 2) * x.pow(2).sum()
                        loss.backward()

                        train_loss += loss.item()
                        opt.step()

                    train_loss /= len(ds)
                    losses = losses.append(pd.DataFrame.from_dict(
                        {'loss': [train_loss],
                         'epoch': [epoch],
                         'lr': [lr],
                         'attempt': [attempt]}), sort=True)
                    pbar.update()
                    pbar.set_description(desc=f'lr = {lr}, attempt = {attempt}, epoch = {epoch}, loss = {train_loss}')
    return losses

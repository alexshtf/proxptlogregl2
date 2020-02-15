import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from convex_losses import LogisticSPPLoss
from convex_on_linear_spp import ConvexOnLinearSPP


def run(dim, ds, epochs, attempts, lrs):
    losses = pd.DataFrame(columns=['lr', 'epoch', 'attempt', 'loss'])
    total_epochs = len(lrs) * len(attempts) * len(epochs)
    with tqdm(total=total_epochs, desc='lr = NA, attempt = NA, epoch = NA, loss = NA', unit='epochs',
              ncols=140) as pbar:
        for lr in lrs:
            for attempt in attempts:
                x = torch.empty(dim, requires_grad=False, dtype=torch.double)
                torch.nn.init.normal_(x)
                opt = ConvexOnLinearSPP(x, lr, LogisticSPPLoss())

                for epoch in epochs:
                    train_loss = 0
                    for x, y in DataLoader(ds, shuffle=True, batch_size=1):
                        xx = x.squeeze(0)

                        if y.item() == 0:
                            a = -xx
                        else:
                            a = xx

                        train_loss += opt.step(a, 0)

                    train_loss /= len(ds)
                    losses = losses.append(pd.DataFrame.from_dict(
                        {'loss': [train_loss],
                         'epoch': [epoch],
                         'lr': [lr],
                         'attempt': [attempt]}), sort=True)
                    pbar.update()
                    pbar.set_description(desc=f'lr = {lr}, attempt = {attempt}, epoch = {epoch}, loss = {train_loss}')
    return losses

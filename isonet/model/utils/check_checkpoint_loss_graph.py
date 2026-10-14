import matplotlib.pyplot as plt 
import matplotlib
import torch
from pathlib import Path
matplotlib.use('Qt5Agg')
from isonet.config import ROOT 


def plot_loss_graph(path: str|Path, title=None, ylable='loss', xlabel='epoch',
                    grid_visible=True, figsize=(10, 8)):
    """path is checkpoint's path"""

    checkpoint = torch.load(path, weights_only=False)
    loss_history = checkpoint['loss_history']
    len_epoch = range(1, 1+len(loss_history['train_loss']))
    if title==None:
        title = f"loss graph of {str(path).replace("\\", "/").split('/')[-1]}"
        
    print(loss_history)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.ylabel(ylable)
    plt.xlabel(xlabel)
    # plt.grid(visible=grid_visible)
    plt.legend('upper left')
    plt.plot(len_epoch, loss_history['train_loss'])
    plt.plot(len_epoch, loss_history['val_loss'])
    plt.show()
    print('aaa')


if __name__ == "__main__":
    plot_loss_graph(ROOT + '/experiment/ex03/output/pretrain_share2/checkpoint1_1/ssl_checkpoint_80epoch.pt')
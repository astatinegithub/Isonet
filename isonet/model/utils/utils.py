import torch
import matplotlib.pyplot as plt 


def save(output_name, is_finetune=True, **kwarg):
    if not(kwarg.get('model', False)): raise KeyError('model parameter is needed')
    if not(kwarg.get('cfg', False)): raise KeyError('cfg parameter is needed')
    if not(kwarg.get('optimizer', False)): raise KeyError('optimizer parameter is needed')
    if not(kwarg.get('loss_history', False)): raise KeyError('loss_history parameter is needed')
    if is_finetune:
        torch.save(
            {
                "encoder": kwarg['model'].encoder.state_dict(),
                "stereo_attention": kwarg['model'].stereo_attention.state_dict(),
                "heads": kwarg['model'].heads.state_dict(),
                "optimizer": kwarg['optimizer'].state_dict(),
                "model": kwarg['model'].state_dict(),
                "epoch": kwarg['epoch'],
                "loss_history": kwarg['loss_history'],
                "cfg": kwarg['cfg']
            },
            output_name
        )
    else:
        torch.save(
            {
                "encoder": kwarg['model'].encoder.state_dict(),
                "atom_head": kwarg['model'].atom_head.state_dict(),
                "bond_head": kwarg['model'].bond_head.state_dict(),
                "optimizer": kwarg['optimizer'].state_dict(),
                "model": kwarg['model'].state_dict(),
                "epoch": kwarg['epoch'],
                "loss_history": kwarg['loss_history'],
                "cfg": kwarg['cfg']
            },
            output_name
        )


def train_val_loss_graph(loss_history: dict[list], save_path=None, dpi=300, plt_close=False):
    epochs = range(1, len(loss_history["train_loss"]) + 1)
    plt.plot(epochs, loss_history['train_loss'], label="Train")
    plt.plot(epochs, loss_history['val_loss'], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    if plt_close:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    ...
    # save("asd", model='model', optimizer='optimizer', epoch='epoch', loss_history='loss_history', cfg='cfg')
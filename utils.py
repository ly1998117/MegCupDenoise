import megengine as meg
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next result to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in meg/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.max_id = str()

    def update(self, val, n=1, pid=''):
        self.val = val
        if val > self.max:
            self.max = val
            self.max_id = pid
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, dir='output', name='checkpoint', best_score=None, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        try:
            os.chmod(self.dir, mode=0o777)
        except Exception:
            pass

    def __call__(self, val_loss, model, optimizer):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            print("Early stop initiated")
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}       best score: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f"{self.best_score} -> {score}")
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # meg.save(model, f'{self.name}_finish_model.pkl')  # 这里会存储迄今最优的模型
        meg.save(
            {
                "state_dict": model.state_dict(),
                "best_loss": self.best_score,
                "best_score": self.best_score,
                "optimizer": optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
            },
            f"{os.path.join(self.dir, self.name)}.pth",
            pickle_protocol=4
        )
        self.val_loss_min = val_loss

    def load_checkpoint(self, model, ignore):
        checkpoint = meg.load(f'{self.dir}/{self.name}.pth')
        model.load_state_dict(checkpoint["state_dict"])
        if not ignore:
            self.best_score = checkpoint["best_score"]
        try:
            print(f"load score: {self.best_score}       lr: {checkpoint['lr']}")
        except KeyError:
            print(f"load score: {self.best_score}")
        model.eval()


def val_predict_plot(model, output_path, metrix, dataloader=None):
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    f = open(os.path.join(output_path, "score.txt"), 'w')
    valid_acc_total = MetricTracker()

    num = dataloader.batch_size * 3
    fout = open(f'data/competition_split_val_prediction.0.2.bin', 'wb')
    with tqdm(dataloader, desc='plot') as bar:
        for i, (img, gt) in enumerate(bar):
            inputs = meg.tensor(img, dtype=np.float32)
            labels = meg.tensor(gt, dtype=np.float32)

            patient_path = os.path.join(output_path, f'{i}.png')

            width = 9
            if num % width == 0:
                height = num // width
            else:
                height = num // width + 1
            plt.figure()
            fig, axs = plt.subplots(height, width, dpi=200, figsize=(width, height))
            axs = axs.reshape(-1)
            [ax.axis('off') for ax in axs]

            idx = 0

            outputs = model(inputs)
            pred = (outputs.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')

            fout.write(pred.tobytes())

            for x, y, g in zip(inputs, outputs, labels):
                ax = axs[idx]
                idx += 1
                ax.imshow((x.squeeze().numpy() * 256).clip(0, 255).astype(np.uint8), cmap='gray')

                ax = axs[idx]
                idx += 1
                ax.imshow((y.squeeze().numpy() * 256).clip(0, 255).astype(np.uint8), cmap='gray')

                ax = axs[idx]
                idx += 1
                ax.imshow((g.squeeze().numpy() * 256).clip(0, 255).astype(np.uint8), cmap='gray')

                score = metrix(outputs, labels)
                f.write(f"id: {i}   score: {score}\n")
                valid_acc_total.update(score, pid=f'{i}')

                plt.suptitle(f"in out gt {i}  {metrix.__name__}: {score}")
                plt.savefig(patient_path)
                plt.close('all')
    fout.close()

    f.write(f'\ntotal dice: {valid_acc_total.avg}\n'
            f'[max dice pid : {valid_acc_total.max_id}      max dice : {valid_acc_total.max}]')
    f.close()

    f = open(os.path.join(output_path, "score.txt"), mode='r')
    dice_data = list(f.readlines())[:-3]
    f.close()
    dice_list = list(map(lambda x: float(x.split()[-1]), dice_data))
    pid = list(map(lambda x: x.split()[1], dice_data))

    x = pid
    y = np.array(dice_list)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x, y, width=.5)
    plt.xticks(rotation=60, fontsize=7)
    path = os.path.join(os.path.split(output_path)[0], 'score.png')
    plt.savefig(path)
    plt.close('all')

import argparse
import random

import megengine as torch
import megengine.module as nn

from tqdm import tqdm
import numpy as np
import os
from data_loader import get_loader, RawUtils
from megengine.utils.module_stats import module_stats
from model.DABSkip1DownSampleGeLU import DAB1skip1DownSampleGeLU

try:
    from utils import AverageValueMeter, EarlyStopping
    from loss import Loss
except ImportError:
    from .utils import AverageValueMeter, EarlyStopping
    from .loss import Loss


def compute_score(p, g):
    if p.shape[1] == 3:
        p = RawUtils.rggb2bayer(p)
        g = RawUtils.rggb2bayer(g)
    pre, gt = p.numpy(), g.numpy()
    means = gt.mean(axis=(2, 3))
    weight = (1 / means) ** 0.5
    diff = np.abs(pre - gt).mean(axis=(2, 3))
    diff = diff * weight
    score = diff.mean()
    score = np.log10(100 / score) * 5
    return score


class Metric(nn.Module):
    __name__ = 'score'

    def __init__(self):
        super(Metric, self).__init__()

    def forward(self, pre, gt):
        return compute_score(pre, gt)


def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s


def run(gm, loader, optimizer, model, loss, metrics, desc):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
    with tqdm(loader, desc=desc) as bar:
        for img, gt in bar:
            data = torch.tensor(img, dtype=np.float32)
            label = torch.tensor(gt, dtype=np.float32)
            with gm:
                optimizer.clear_grad()
                y_pred = model(data)
                ls = loss(y_pred, label)

                if desc == 'train':
                    gm.backward(ls)
                    optimizer.step()
            loss_value = ls.numpy()
            loss_meter.add(loss_value)
            loss_logs = {loss.__name__: loss_meter.mean}
            logs.update(loss_logs)
            for metric_fn in metrics:
                metric_value = metric_fn(y_pred, label)
                metrics_meters[metric_fn.__name__].add(metric_value)

            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
            logs.update(metrics_logs)
            s = format_logs(logs)
            bar.set_postfix_str(s)
    return logs


def train(name, model, resume=False, patience=20, loss=None, epochs=10, lr=2e-4, aug=False, finetune=False,
          mode='bayer', bayerAug=False):
    os.makedirs(name, exist_ok=True, mode=0o777)
    try:
        os.chmod(name, mode=0o777)
    except Exception:
        pass
    train_loader = get_loader('train', aug, finetune, batchsz=batchsz, mode=mode, bayerAug=bayerAug)
    val_loader = get_loader('val', False, False, batchsz=16, mode=mode)
    print(f'train batch: {len(train_loader)}     val batch: {len(val_loader)}')
    gm = torch.autodiff.GradManager().attach(model.parameters())

    optimizer = torch.optimizer.Adam(params=model.parameters(), lr=lr)
    stopper = EarlyStopping(dir=name, name=name, patience=patience)

    if resume:
        stopper.load_checkpoint(model, args.ignore)
    metrics = [Metric()]
    train_logs_list, valid_logs_list = [], []

    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g['lr'] = lr * (1 - (epoch / epochs) ** 0.9)

        print(f"\nEpoch: {epoch}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, name: {name}")

        model.train()
        train_logs = run(gm, train_loader, optimizer, model, loss, metrics, 'train')
        train_logs_list.append(train_logs)

        model.eval()
        valid_logs = run(gm, val_loader, optimizer, model, loss, metrics, 'val')
        stopper(valid_logs[metrics[0].__name__], model, optimizer)
        valid_logs_list.append(valid_logs)

        if stopper.early_stop:
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="./checkpoint", type=str, help="checkpoint name")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=400)

    parser.add_argument("--bz", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)

    parser.add_argument('--aug', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--ignore', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    patchsz = 256
    batchsz = args.bz
    torch.set_default_device(f'gpu{args.device}')
    random.seed(41)
    input_data = np.random.rand(1, 1, 256, 256).astype("float32")

    model = DAB1skip1DownSampleGeLU()

    total_stats, stats_details = module_stats(
        model,
        inputs=(input_data),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=False,
    )

    print("params %.3fK MAC/pixel %.0f" % (
        total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))
    loss = Loss()
    resume = args.resume
    aug = args.aug
    train(args.name, model, resume, loss=loss, epochs=args.epoch, patience=args.epoch, lr=args.lr, aug=aug,
          mode='bayer', bayerAug=False, finetune=True)

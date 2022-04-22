import megengine.functional as F


def compute_score_loss(pre, gt):
    means = gt.mean(axis=(2, 3))
    weight = (1 / means) ** 0.5
    diff = F.abs(pre - gt).mean(axis=(2, 3))
    diff = diff * weight
    score = diff.mean()
    score = F.log(100 / score) * 4
    return 1 / score


class Loss:
    __name__ = 'Loss'

    def __init__(self):
        super(Loss, self).__init__()
        self.l1 = F.nn.l1_loss

    def __call__(self, pre, gt):
        return 256 * self.l1(pre, gt) + 10 * compute_score_loss(pre, gt)

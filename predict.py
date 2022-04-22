from utils import EarlyStopping
from tqdm import tqdm
from train import compute_score
import numpy as np
import megengine as meg
import os
import matplotlib.pyplot as plt
import cv2


def linear_expand(x):
    x = (x - x.min()) / (x.max() - x.min()) * 256
    return x


def predict(name, model, load=False, batch_size=16):
    print('prediction')
    if load:
        stopper = EarlyStopping(dir=name, name=name, patience=10)
        stopper.load_checkpoint(model)
    content = open('data/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    fout = open(name + '/result.bin', 'wb')
    os.makedirs(os.path.join(name, 'plot'), exist_ok=True)
    try:
        os.chmod(os.path.join(name, 'plot'), mode=0o777)
    except Exception:
        pass
    num = batch_size * 2
    for i in tqdm(range(0, len(samples_ref), batch_size)):
        patient_path = os.path.join(name, 'plot', f'{i}.png')
        i_end = min(i + 16, len(samples_ref))
        batch_inp = meg.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))

        pred = model(batch_inp)

        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        batch_inp = (batch_inp.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

        width = 8
        if num % width == 0:
            height = num // width
        else:
            height = num // width + 1
        plt.figure()
        fig, axs = plt.subplots(height, width, dpi=300, figsize=(width, height))
        axs = axs.reshape(-1)
        [ax.axis('off') for ax in axs]

        idx = 0
        for x, y in zip(batch_inp, pred):
            ax = axs[idx]
            idx += 1
            ax.imshow(
                (linear_expand(cv2.cvtColor(x, cv2.COLOR_BAYER_RGGB2RGB).astype(np.float32))).clip(0, 255).astype(
                    np.uint8))

            ax = axs[idx]
            idx += 1
            ax.imshow(
                (linear_expand(cv2.cvtColor(y, cv2.COLOR_BAYER_RGGB2RGB).astype(np.float32))).clip(0, 255).astype(
                    np.uint8))

        plt.suptitle(f"in out gt {i}")
        plt.savefig(patient_path)
        plt.close('all')
    fout.close()


if __name__ == '__main__':
    from model.DAB1SumSkip1DownSampleGeLU import DAB1skip1DownSampleGeLU

    meg.set_default_device('gpux')
    model = DAB1skip1DownSampleGeLU()
    predict('DABWith_1_Skip_1_DownSample_GeLU_AUG', model, True)

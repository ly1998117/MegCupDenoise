import numpy as np
import os, cv2, random
from megengine.data.dataset import Dataset
from megengine.data import SequentialSampler, DataLoader, RandomSampler


class Split:
    @staticmethod
    def load_data():
        print('loading data')
        samples_ref = np.frombuffer(open('data/competition_train_input.0.2.bin', 'rb').read(),
                                    dtype='uint16').reshape((-1, 256, 256))
        samples_gt = np.frombuffer(open('data/competition_train_gt.0.2.bin', 'rb').read(),
                                   dtype='uint16').reshape((-1, 256, 256))
        samples_test = np.frombuffer(open('data/competition_test_input.0.2.bin', 'rb').read(),
                                     dtype='uint16').reshape((-1, 256, 256))

        return samples_ref, samples_gt, samples_test

    @staticmethod
    def split():
        os.makedirs('data', exist_ok=True)
        samples_ref, samples_gt, samples_test = Split.load_data()
        C = samples_ref.shape[0]
        TrainC = int(0.8 * C)
        Split.save2bin('train', samples_ref[0:TrainC], samples_gt[0:TrainC])
        Split.save2bin('val', samples_ref[TrainC:], samples_gt[TrainC:])
        fout = open(f'data/competition_test_input.0.2.bin', 'wb')
        fout.write((samples_test).clip(0, 65535).astype('uint16').tobytes())
        fout.close()

    @staticmethod
    def save2bin(name, inputs, gt):
        print(name)
        fout = open(f'data/competition_split_{name}_input.0.2.bin', 'wb')
        fgt = open(f'data/competition_split_{name}_gt.0.2.bin', 'wb')

        fout.write((inputs).clip(0, 65535).astype('uint16').tobytes())
        fgt.write((gt).clip(0, 65535).astype('uint16').tobytes())
        fout.close()
        fgt.close()
        print('done')


class RawUtils:

    @classmethod
    def grbg2rggb(cls, grbg):
        rggb = grbg[[1, 0, 3, 2], :]
        return rggb

    @classmethod
    def gbrg2rggb(cls, gbrg):
        return gbrg[[2, 3, 0, 1], :]

    @classmethod
    def bayer2rggb(cls, bayer):
        H, W = bayer.shape[-2], bayer.shape[-1]
        if bayer.ndim == 2:
            bayer = bayer.reshape(H // 2, 2, W // 2, 2).transpose(1, 3, 0, 2).reshape(4, H // 2, W // 2)
        elif bayer.ndim == 4:
            bayer = bayer.reshape(-1, 1, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4).reshape(-1, 4, H // 2,
                                                                                                   W // 2)
        return bayer

    @classmethod
    def rggb2bayer(cls, rggb):
        H, W = rggb.shape[-2], rggb.shape[-1]
        if rggb.ndim == 3:
            rggb = rggb.reshape(2, 2, H, W).transpose(2, 0, 3, 1).reshape(H * 2, W * 2)
        elif rggb.ndim == 4:
            rggb = rggb.reshape(-1, 2, 2, H, W).transpose(0, 3, 1, 4, 2).reshape(-1, 1, H * 2, W * 2)
        return rggb

    @classmethod
    def bayer2rgb(cls, *bayer_01s, wb_gain, CCM, gamma=2.2):
        wb_gain = np.array(wb_gain)[[0, 1, 1, 2]]
        res = []
        for bayer_01 in bayer_01s:
            bayer = cls.rggb2bayer(
                (cls.bayer2rggb(bayer_01) * wb_gain).clip(0, 1)
            ).astype(np.float32)
            bayer = np.round(np.ascontiguousarray(bayer) * 65535).clip(0, 65535).astype(np.uint16)
            rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB_EA).astype(np.float32) / 65535
            rgb = rgb.dot(np.array(CCM).T).clip(0, 1)
            rgb = rgb ** (1 / gamma)
            res.append(rgb.astype(np.float32))

        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def flip(cls, x, axis, bayer_aug=False):
        if axis == 0:
            if bayer_aug:
                return cls.bayer_aug(x, True, False, False, 'rggb')
            else:
                f = np.flip(x, axis=0)
                gbrg = cls.bayer2rggb(f)
                rggb = cls.gbrg2rggb(gbrg)
                return cls.rggb2bayer(rggb)
        if axis == 1:
            if bayer_aug:
                return cls.bayer_aug(x, False, True, False, 'rggb')
            else:
                f = np.flip(x, axis=1)
                grbg = cls.bayer2rggb(f)
                rggb = cls.grbg2rggb(grbg)
                return cls.rggb2bayer(rggb)
        if axis == 2:
            if bayer_aug:
                return cls.bayer_aug(x, False, False, True, 'rggb')
            else:
                x = cls.flip(x, 0, False)
                return cls.flip(x, 1, False)

    @classmethod
    def bayer_unify(cls, raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
        """
        Convert a bayer raw image from one bayer pattern to another.

        Parameters
        ----------
        raw : np.ndarray in shape (H, W)
            Bayer raw image to be unified.
        input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
            The bayer pattern of the input image.
        target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
            The expected output pattern.
        mode: {"crop", "pad"}
            The way to handle submosaic shift. "crop" abandons the outmost pixels,
            and "pad" introduces extra pixels. Use "crop" in training and "pad" in
            testing.
        """
        BAYER_PATTERNS = ["rggb", "bggr", "grbg", "gbrg"]
        NORMALIZATION_MODE = ["crop", "pad"]

        if input_pattern not in BAYER_PATTERNS:
            raise ValueError('Unknown input bayer pattern!')
        if target_pattern not in BAYER_PATTERNS:
            raise ValueError('Unknown target bayer pattern!')
        if mode not in NORMALIZATION_MODE:
            raise ValueError('Unknown normalization mode!')
        if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
            raise ValueError('raw should be a 2-dimensional numpy.ndarray!')

        if input_pattern == target_pattern:
            h_offset, w_offset = 0, 0
        elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
            h_offset, w_offset = 1, 0
        elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
            h_offset, w_offset = 0, 1
        elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
            h_offset, w_offset = 1, 1
        else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
            raise RuntimeError('Unexpected pair of input and target bayer pattern!')

        if mode == "pad":
            out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
            if h_offset == 1:
                out = out[2:, :]
            if w_offset == 1:
                out = out[:, 2:]
        elif mode == "crop":
            h, w = raw.shape
            out = raw[h_offset:h - h_offset, w_offset:w - w_offset]
        else:
            raise ValueError('Unknown normalization mode!')

        return out

    @classmethod
    def bayer_aug(cls, raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool, input_pattern: str) -> np.ndarray:
        """
        Apply augmentation to a bayer raw image.

        Parameters
        ----------
        raw : np.ndarray in shape (H, W)
            Bayer raw image to be augmented. H and W must be even numbers.
        flip_h : bool
            If True, do vertical flip.
        flip_w : bool
            If True, do horizontal flip.
        transpose : bool
            If True, do transpose.
        input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
            The bayer pattern of the input image.
        """
        BAYER_PATTERNS = ["rggb", "bggr", "grbg", "gbrg"]
        if input_pattern not in BAYER_PATTERNS:
            raise ValueError('Unknown input bayer pattern!')
        if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
            raise ValueError('raw should be a 2-dimensional numpy.ndarray')
        if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
            raise ValueError('raw should have even number of height and width!')

        aug_pattern, target_pattern = input_pattern, input_pattern

        out = raw
        if flip_h:
            out = out[::-1, :]
            aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
        if flip_w:
            out = out[:, ::-1]
            aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
        if transpose:
            out = out.T
            aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

        out = cls.bayer_unify(out, aug_pattern, target_pattern, "pad")
        return out


class BasicDataset(Dataset):
    def __init__(self, path=None, aug=None, look=False, mode='bayer', bayerAug=False):
        super(BasicDataset, self).__init__()
        self.aug = aug
        self.look = look
        self.path = path
        self.data = self.load()
        self.mode = mode
        self.bayerAug = bayerAug

    def __len__(self):
        return len(self.data[1])

    def preprocess(self, img, gt):
        img = img / 256
        gt = gt / 256

        if not self.look:
            img = img / 256
            gt = gt / 256

        if self.aug:
            img, gt = random_augmentation(img, gt, self.bayerAug)
        if self.mode == 'bayer':
            img = np.expand_dims(img, 0)
            gt = np.expand_dims(gt, 0)
        img = img.astype(np.float32)
        gt = gt.astype(np.float32)
        return img, gt

    def load(self):
        img_path, gt_path = self.path
        with open(img_path, 'rb') as f:
            img = np.frombuffer(f.read(), dtype='uint16').reshape(-1, 256, 256).astype(np.float32)

        with open(gt_path, 'rb') as f:
            gt = np.frombuffer(f.read(), dtype='uint16').reshape(-1, 256, 256).astype(np.float32)
        return img, gt

    def __getitem__(self, idx):
        img, gt = self.data[0][idx], self.data[1][idx]
        img, gt = self.preprocess(img, gt)
        if self.mode == 'rggb':
            img = RawUtils.bayer2rggb(img)
            gt = RawUtils.bayer2rggb(gt)
        return img, gt


def get_path(name, finetune):
    if finetune:
        return f'data/competition_train_input.0.2.bin', f'data/competition_train_gt.0.2.bin'
    return f'data/competition_split_{name}_input.0.2.bin', f'data/competition_split_{name}_gt.0.2.bin'


def get_loader(name, aug, finetune, batchsz=16, mode='bayer', bayerAug=False):
    dataset = BasicDataset(path=get_path(name, finetune), aug=aug, mode=mode, bayerAug=bayerAug)
    if name == 'val':
        sampler = SequentialSampler(dataset, batch_size=batchsz)
    else:
        sampler = RandomSampler(dataset, batch_size=batchsz)
    dataloader = DataLoader(dataset, sampler=sampler, num_workers=4)
    return dataloader


class DataAugmentation:
    @classmethod
    def flip(cls, image, mode, bayerAug):
        if mode == 0:
            out = image
        elif mode == 1:
            out = RawUtils.flip(image, 1, bayerAug)
        elif mode == 2:
            out = RawUtils.flip(image, 0, bayerAug)
        elif mode == 3:
            out = RawUtils.flip(image, 2, bayerAug)
        else:
            raise Exception('Invalid choice of image transformation')
        return out

    @classmethod
    def rotate(cls, image, mode, bayerAug=False):
        if mode == 0:
            # original
            out = image
        elif mode == 1:
            if bayerAug:
                out = RawUtils.bayer_unify(np.rot90(image, k=1), 'gbrg', 'rggb', 'pad')
            else:
                rggb = RawUtils.bayer2rggb(image)
                out = np.rot90(rggb, axes=(1, 2))
                out = RawUtils.rggb2bayer(out)
        elif mode == 2:
            # rotate 180 degree
            if bayerAug:
                out = RawUtils.bayer_unify(np.rot90(image, k=2), 'bggr', 'rggb', 'pad')
            else:
                rggb = RawUtils.bayer2rggb(image)
                out = np.rot90(rggb, k=2, axes=(1, 2))
                out = RawUtils.rggb2bayer(out)
        elif mode == 3:
            # rotate 270 degree
            if bayerAug:
                out = RawUtils.bayer_unify(np.rot90(image, k=3), 'grbg', 'rggb', 'pad')
            else:
                rggb = RawUtils.bayer2rggb(image)
                out = np.rot90(rggb, k=3, axes=(1, 2))
                out = RawUtils.rggb2bayer(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out


def random_augmentation(img, gt, bayerAug=False):
    if random.randint(0, 1) == 1:
        img, gt = img.copy(), gt.copy()
        flag_aug = random.randint(0, 3)
        img, gt = DataAugmentation.flip(img, flag_aug, bayerAug=bayerAug), DataAugmentation.flip(gt, flag_aug,
                                                                                                 bayerAug=bayerAug)

        flag_aug = random.randint(0, 3)
        img, gt = DataAugmentation.rotate(img, flag_aug, bayerAug=bayerAug), DataAugmentation.rotate(gt, flag_aug,
                                                                                                     bayerAug=bayerAug)

    return img, gt


def random_noise_change(img, gt):
    if random.randint(0, 1) == 2:
        img, gt = img.copy(), gt.copy()
        noise = img - gt
        np.random.shuffle(noise)
        img = gt + noise
    return img, gt


class BayerUnify:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    samples_ref, samples_gt, samples_test = Split.load_data()
    raw = RawUtils

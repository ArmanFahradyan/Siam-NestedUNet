import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

import cv2
import random


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return {'image': (img1, img2),
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img1 = img1.rotate(rotate_degree, Image.BILINEAR)
        img2 = img2.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2),
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}


class RandomProjectiveTransformation(object):
    def __init__(self,
                 angle_range=[-0.03, 0.03],
                 scale_range=[1.04, 1.07],
                 # v_range=[-0.5, 0.5],
                 # v1_range=[-0.5, 0.5],
                 # v2_range=[-0.5, 0.5],
                 lyambda_range=[0.98, 1.02]):
        self.angle_range = angle_range
        self.scale_range = scale_range
        # self.v_range = v_range
        # self.v1_range = v1_range
        # self.v2_range = v2_range
        self.lyambda_range = lyambda_range

    def cut(self, image, diff_mask, soft=True):
        mask = (image == 0).all(axis=2)
        # maaaaaask = mask * 255
        # cv2.imwrite("./testing_dir/maaaaaask.jpg", maaaaaask)
        mask_col = mask.all(axis=0)
        mask_row = mask.all(axis=1)
        row_start = np.argmin(mask_row)
        row_finish = -np.argmin(mask_row[::-1])
        col_start = np.argmin(mask_col)
        col_finish = -np.argmin(mask_col[::-1])
        cropped_image = image[row_start:row_finish, col_start:col_finish, :]
        cropped_mask = diff_mask[row_start:row_finish, col_start:col_finish]
        if soft:
            return cropped_image, cropped_mask
        mask = (cropped_image == 0).all(axis=2)
        # maaaaaask = mask * 255
        # cv2.imwrite("./testing_dir/maaaaaask.jpg", maaaaaask)
        r1 = 0
        r2 = -1
        c1 = 0
        c2 = -1
        while np.sum(mask[r1]) >= 20:  # mask[r1, 0] and mask[r1, -1]:
            r1 += 1
        while np.sum(mask[r2]) >= 20:  # mask[r2, 0] and mask[r2, -1]:
            r2 -= 1
        while np.sum(mask[:, c1]) >= 20:  # mask[0, c1] and mask[-1, c1]:
            c1 += 1
        while np.sum(mask[:, c2]) >= 20:  # mask[0, c2] and mask[-1, c2]:
            c2 -= 1
        # c1 = np.argmin(mask[0])
        # c2 = np.argmin(mask[-1])
        # r1 = np.argmin(mask[:, 0])
        # r2 = np.argmin(mask[:, -1])
        # print(r1, r2)
        # print(c1, c2)
        # mask_col = mask.any(axis=0)
        # mask_row = mask.any(axis=1)
        # row_start = np.argmin(mask_row)
        # row_finish = -np.argmin(mask_row[::-1])
        # col_start = np.argmin(mask_col)
        # col_finish = -np.argmin(mask_col[::-1])

        # row_start, row_finish = min(r1, r2), max(r1, r2)+1
        # col_start, col_finish = min(c1, c2), max(c1, c2) + 1
        row_start, row_finish = r1, r2
        col_start, col_finish = c1, c2
        return cropped_image[row_start:row_finish, col_start:col_finish, :], cropped_mask[row_start:row_finish, col_start:col_finish]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.25:
            return {'image': (img1, img2),
                    'label': mask}

        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        mask = np.array(mask)
        # h, w, _ = img2.shape
        # img_src_coordinate = np.array([[0, 0], [0, h], [w, 0], [w, h]])

        angle = random.uniform(*self.angle_range)
        scale = random.uniform(*self.scale_range)
        # v = random.uniform(*self.v_range)
        v = 1.02
        # v1 = random.uniform(*self.v1_range)
        v1 = 1e-04
        # v2 = random.uniform(*self.v2_range)
        v2 = 1e-04
        lyambda = random.uniform(*self.lyambda_range)
        if lyambda == 0:
            lyambda = 1

        v_matrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [v1, v2, v]])

        l_matrix = np.array([[lyambda, 0, 0],
                             [0, 1/lyambda, 0],
                             [0, 0, 1]])

        r_s_matrix = np.array([[scale*np.cos(angle), -scale*np.sin(angle), 15],
                               [scale*np.sin(angle), scale*np.cos(angle), 15],
                               [0, 0, 1]])

        matrix = r_s_matrix @ l_matrix @ v_matrix

        # paste_coordinate = np.array([[2, 2], [3, h-1], [w+2, 1], [w, h]])
        # matrix, _ = cv2.findHomography(img_src_coordinate, paste_coordinate, 0)
        # print(matrix)
        # print(matrix.shape)
        img2 = cv2.warpPerspective(img2, matrix, (300, 300))
        mask = cv2.warpPerspective(mask, matrix, (300, 300))
        # cv2.imwrite("./testing_dir/chemmanuminch.jpg", img2)
        # cv2.imwrite("./testing_dir/mask_chemmanuminch.jpg", mask)

        # mask_col = (img2 == 0).all(axis=0)[:, 0]
        # mask_row = (img2 == 0).all(axis=1)[:, 0]
        # row_start = np.argmin(mask_row)
        # row_finish = -np.argmin(mask_row[::-1])
        # col_start = np.argmin(mask_col)
        # col_finish = -np.argmin(mask_col[::-1])
        #
        # print(row_start, row_finish)
        # print(col_start, col_finish)
        #
        # cropped_img = img2.copy()[row_start:row_finish, col_start:col_finish, :]


        # cropped_img_soft, cropped_mask_soft = self.cut(image=img2, diff_mask=mask, soft=True)
        cropped_img_hard, cropped_mask_hard = self.cut(image=img2, diff_mask=mask, soft=False)

        # cropped_mask_hard = self.cut(image=mask, soft=True)
        # print("soft:", cropped_img_soft.shape)
        # print("hard:", cropped_img_hard.shape)
        # cv2.imwrite("./testing_dir/soft_ktrats_chemmanuminch.jpg", cropped_img_soft)
        # cv2.imwrite("./testing_dir/hard_ktrats_chemmanuminch.jpg", cropped_img_hard)
        # cv2.imwrite("./testing_dir/soft_ktrats_mask.jpg", cropped_mask_soft)
        # cv2.imwrite("./testing_dir/hard_ktrats_mask.jpg", cropped_mask_hard)


        # cropped_img_hard = cv2.resize(cropped_img_hard, (256, 256))
        # cropped_mask_hard = cv2.resize(cropped_mask_hard, (256, 256))

        # cv2.imwrite("./testing_dir/hard_ktrats_chemmanuminch.jpg", cropped_img_hard)
        # cv2.imwrite("./testing_dir/hard_ktrats_mask.jpg", cropped_mask_hard)

        # ------------------------

        # assert 1 == 0
        #
        # r_zeros = 256 - cropped_img_hard.shape[0]
        # c_zeros = 256 - cropped_img_hard.shape[1]
        #
        # tmp = np.concatenate([cropped_img_hard, np.zeros((r_zeros, cropped_img_hard.shape[1], 3))], axis=0)
        # cropped_img_hard = np.concatenate([tmp, np.zeros((256, c_zeros, 3))], axis=1)
        # assert cropped_img_hard.shape == (256, 256, 3)
        #
        # tmp = np.concatenate([cropped_mask_hard, np.zeros((r_zeros, cropped_img_hard.shape[1]))], axis=0)
        # cropped_mask_hard = np.concatenate([tmp, np.zeros((256, c_zeros))], axis=1)
        # assert cropped_mask_hard.shape == (256, 256)

        # ------------------------

        cropped_img_hard = cv2.cvtColor(cropped_img_hard, cv2.COLOR_BGR2RGB)
        cropped_img_hard = Image.fromarray(cropped_img_hard)
        cropped_mask_hard = Image.fromarray(cropped_mask_hard)

        cropped_img_hard = cropped_img_hard.resize((256, 256), Image.BILINEAR)
        cropped_mask_hard = cropped_mask_hard.resize((256, 256), Image.NEAREST)

        # print(type(cropped_img_hard))

        # print("finish")

        return {'image': (img1, cropped_img_hard),
                'label': cropped_mask_hard}


'''
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
'''
train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            # RandomProjectiveTransformation(),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

test_transforms = transforms.Compose([
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomFixRotate(),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

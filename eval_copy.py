import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# test_loader = get_test_loaders(opt)

path = './weights/snunet-32.pt'   # the path of the model
model = torch.load(path)

# _names = ["IMG_20220824_163932.jpg", "IMG_20220824_163938.jpg"]

_paths = ["/home/user/Downloads/Telegram Desktop/IMG_20220824_163932.jpg", "/home/user/Downloads/Telegram Desktop/IMG_20220824_163938.jpg"]
# ["./ChangeDetectionDataset/Real/subset/test/A/00000.jpg", "./ChangeDetectionDataset/Real/subset/test/B/00000.jpg"]

image1 = cv2.imread(_paths[0])  # cv2.imread(f"/home/user/Downloads/Telegram Desktop/{_names[0]}")
image2 = cv2.imread(_paths[1])  # cv2.imread(f"/home/user/Downloads/Telegram Desktop/{_names[1]}")

image1 = cv2.resize(image1, (256, 256))  # , interpolation = cv2.INTER_AREA)
image2 = cv2.resize(image2, (256, 256))

cv2.imwrite("./testing/img1.png", image1)
cv2.imwrite("./testing/img2.png", image2)

image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
image1 = np.moveaxis(image1, -1, 0)
image1 = image1[np.newaxis, :]
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
image2 = np.moveaxis(image2, -1, 0)
image2 = image2[np.newaxis, :]

img1 = torch.tensor(image1)
img2 = torch.tensor(image2)

print(img1.shape)

model.eval()

# _names = [i for i in os.listdir('./ChangeDetectionDataset/Real/subset/test/A/') if not i.startswith('.')]

with torch.no_grad():
    img1 = img1.float().to(dev)
    img2 = img2.float().to(dev)

    cd_preds = model(img1, img2)
    
    cd_preds = cd_preds[-1]
    print("#", torch.sum(cd_preds == 0))
    print(cd_preds.shape)
    _, cd_preds = torch.max(cd_preds, dim=1)
    # cd_preds = cd_preds[0][0]
    print("@", torch.sum(cd_preds == 0))

    print(cd_preds.shape)
    print(cd_preds)

    img = cd_preds.data.cpu().numpy().squeeze() * 255
    file_path = './testing/' + "prediction"
    cv2.imwrite(file_path + '.png', img)

    # print(labels.shape)

    # for i in range(cd_preds.shape[0]):
    #     img = cd_preds[i].data.cpu().numpy().squeeze() * 255
    #     file_path = './testing/' + "prediction"
    #     cv2.imwrite(file_path + '.png', np.moveaxis(img, 0, -1))


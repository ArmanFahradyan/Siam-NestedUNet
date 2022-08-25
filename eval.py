import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = './weights/snunet-32.pt'   # the path of the model
model = torch.load(path)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

_names = [i for i in os.listdir('./ChangeDetectionDataset/Real/subset/test/A/') if not i.startswith('.')]

with torch.no_grad():
    tbar = tqdm(test_loader)
    ind = 0
    for (batch_img1, batch_img2, labels) in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten()).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

        # print(labels.shape)

        for i in range(cd_preds.shape[0]):
            img = cd_preds[i].data.cpu().numpy().squeeze() * 255
            label = labels[i].data.cpu().numpy().squeeze() * 255
            file_path = './output_images/' + str(_names[ind*cd_preds.shape[0] + i])
            file_path_label = './output_images/' + str(_names[ind*cd_preds.shape[0] + i] + "L")
            cv2.imwrite(file_path + '.png', img)
            cv2.imwrite(file_path_label + '.png', label)

        ind += 1

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))

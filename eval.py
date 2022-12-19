import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np

# output_dir = './output_images_CDD_256_original_weights/'

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

model = torch.load(opt.model_path)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

_names_path = opt.dataset_dir+"test/A/"
_names = [i for i in os.listdir(_names_path) if not i.startswith('.')]
_names.sort()

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

        ans_conf_mat = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten()).ravel()

        if len(ans_conf_mat) == 1:
            tn, fp, fn, tp = ans_conf_mat[0], 0, 0, 0
        else:
            tn, fp, fn, tp = ans_conf_mat

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

        for i in range(cd_preds.shape[0]):
            img = cd_preds[i].data.cpu().numpy().squeeze() * 255
            label = labels[i].data.cpu().numpy().squeeze() * 255
            file_path = opt.output_dir + str(_names[ind*opt.batch_size + i]) # cd_preds.shape[0]
            cv2.imwrite(file_path, np.concatenate([label, 123*np.ones((label.shape[0], 5)), img], axis=1))

        ind += 1

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))

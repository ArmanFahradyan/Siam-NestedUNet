import torch.utils.data
import torch
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np
from divide_and_detect import divide_and_detect
from utils.parser import get_parser_with_args

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load(opt.model_path, map_location=dev)

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

file_names = [i for i in os.listdir(opt.dataset_dir+'val/A/') if not i.startswith('.')]
file_names.sort()

for file_name in file_names:
    file_path_1 = opt.dataset_dir+'val/A/' + file_name
    file_path_2 = opt.dataset_dir+'val/B/' + file_name
    label_path = opt.dataset_dir+'val/OUT/' + file_name
    label_map = cv2.imread(label_path)
    label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2GRAY)
    predict_map = divide_and_detect(file_path_1, file_path_2, destination_path=None, store_image=False)

    tn, fp, fn, tp = confusion_matrix(label_map.flatten(), predict_map.flatten()).ravel()

    c_matrix['tn'] += tn
    c_matrix['fp'] += fp
    c_matrix['fn'] += fn
    c_matrix['tp'] += tp

    cv2.imwrite(opt.output_dir+file_name, np.concatenate([label_map, 123*np.ones((label_map.shape[0], 5)), predict_map], axis=1))


tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))




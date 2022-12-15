from pickletools import uint8
from utils.parser import get_parser_with_args
from sliding_window_helpers import gkern, get_weights, make_generators, sliding_window_predict, compute_c_matrix
import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
import time

parser, metadata = get_parser_with_args(metadata_json='./metadata_sliding_window.json', 
                                        description='Evaluating model with sliding window method')
opt = parser.parse_args()

_names = [i.split('/')[-1] for i in os.listdir(opt.dataset_dir+'test/A/') if not i.startswith('.')]

_names.sort()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(opt.model_path, map_location='cpu')

model.to(dev)

gaussian = gkern(opt.kernel_size, opt.sigma)

# t1 = time.time()

weights = get_weights(opt.image_shape, opt.kernel_size, gaussian, opt.stride)

generators = make_generators(weights)

# lens = []
# for ii in range(weights.shape[0]):
#     for jj in range(weights.shape[1]):
#         lens.append(len(weights[ii, jj]))
# print("max len:", max(lens))

# input(time.time()-t1)

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

for name in tqdm(_names):
    img1 = cv2.imread(opt.dataset_dir+'test/A/'+name)
    img2 = cv2.imread(opt.dataset_dir+'test/B/'+name)
    label_map = cv2.imread(opt.dataset_dir+'test/OUT/'+name)
    predict_map = sliding_window_predict(img1, img2, model, opt.kernel_size, opt.stride, generators)
    predict_map = np.rint(predict_map).astype('uint8')
    label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2GRAY)
    predict_map = cv2.cvtColor(predict_map, cv2.COLOR_BGR2GRAY)
    predict_map[predict_map >= 0.8*255] = 255
    predict_map[predict_map < 0.2*255] = 0
    tn, fp, fn, tp = compute_c_matrix(label_map, predict_map)
    c_matrix['tn'] += tn
    c_matrix['fp'] += fp
    c_matrix['fn'] += fn
    c_matrix['tp'] += tp
    cv2.imwrite(f"{opt.output_dir}{name}", np.concatenate([label_map, 123*np.ones((label_map.shape[0], 5)), predict_map], axis=1))

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
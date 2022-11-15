import torch.utils.data
import torch
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy as np
from divide_and_detect import divide_and_detect

if not os.path.exists('./output_images_EBDD'):
    os.mkdir('./output_images_EBDD')

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = './tmp1/checkpoint_epoch_199.pt'  # './weights/snunet-32.pt' # the path of the model
data_path = './data_512/'
model = torch.load(model_path, map_location=dev)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

file_names = [i for i in os.listdir(data_path+'val/A/') if not i.startswith('.')]
file_names.sort()

for file_name in file_names:
    file_path_1 = data_path+'val/A/' + file_name
    file_path_2 = data_path+'val/B/' + file_name
    label_path = data_path+'val/OUT/' + file_name
    label_map = cv2.imread(label_path)
    label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2GRAY)
    assert label_map.shape == (512, 512)
    predict_map = divide_and_detect(file_path_1, file_path_2, destination_path=None, store_image=False)

    tn, fp, fn, tp = confusion_matrix(label_map.flatten(), predict_map.flatten()).ravel()

    c_matrix['tn'] += tn
    c_matrix['fp'] += fp
    c_matrix['fn'] += fn
    c_matrix['tp'] += tp

    cv2.imwrite('./output_images_EBDD/'+file_name, np.concatenate([label_map, 123*np.ones((label_map.shape[0], 5)), predict_map], axis=1))


tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))




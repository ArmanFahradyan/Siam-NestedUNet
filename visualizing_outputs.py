import cv2
import os
import numpy as np

predictions_256_path = './output_images1/'
predictions_divided_path = './output_images_EBDD/'
images_path = './data_512/val/'
output_dir = './visualized_outputs/'

if not os.path.exists('./visualized_outputs'):
    os.mkdir('./visualized_outputs')

_names = [i for i in os.listdir(predictions_256_path) if not i.startswith('.')]
_names.sort()

for name in _names:
    img1 = cv2.imread(images_path+'A/'+name[:-4])
    img2 = cv2.imread(images_path+'B/'+name[:-4])
    label_pred_divided = cv2.imread(predictions_divided_path+name[:-4])
    label = label_pred_divided[:, :512, :]
    pred_divided = label_pred_divided[:, -512:, :]
    label_pred_256 = cv2.imread(predictions_256_path+name)
    pred_256 = label_pred_256[:, -256:, :]
    pred_512 = cv2.resize(pred_256, (512, 512))
    upper_part = np.concatenate([np.zeros((512, 258, 3)), img1, 123*np.ones((512, 6, 3)), img2, np.zeros((512, 258, 3))], axis=1)
    # print(label.shape, pred_512.shape, pred_divided.shape)
    lower_part = np.concatenate([label, 123*np.ones((512, 5, 3)), pred_512, 123*np.ones((512, 5, 3)), pred_divided], axis=1)
    # lower_part = np.concatenate([(lower_part,)*3], axis=2)
    overall_image = np.concatenate([upper_part, lower_part], axis=0)
    cv2.imwrite(output_dir+name, overall_image)
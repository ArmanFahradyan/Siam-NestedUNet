import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-pd', "--dataset_dir", type=str, help=r"path of the dataset until {train, val, test} directories (not including)")
parser.add_argument('-od', "--output_dir", type=str, help="output directory")
parser.add_argument('-lp', "--list_paths", type=str, nargs='+', help="the list of different methods' outputs' paths")
parser.add_argument('-ln', "--list_names", type=str, nargs='+', help="the list of different methods' names")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

_names = [i for i in os.listdir(args.dataset_dir+"test/A/") if not i.startswith('.')]
_names.sort()

for name in tqdm(_names):
    img1 = cv2.imread(args.dataset_dir+"test/A/"+name)
    img2 = cv2.imread(args.dataset_dir+"test/B/"+name)
    label = cv2.imread(args.dataset_dir+"test/OUT/"+name)
    
    assert img2.shape[0] == img1.shape[0], "inappropriate image shape"
    assert label.shape[0] == img1.shape[0], "inappropriate label shape"

    if name == _names[0]:
        assert len(args.list_paths) == len(args.list_names), "inappropriate number of methods or names"
        num_methods = len(args.list_paths)
        img_size = img1.shape[0]
        row_count = (len(args.list_paths)+2) // 3
        col_stick = 123*np.ones((img_size, 5, 3))
        row_stick = 123*np.ones((5, 3*img_size + 2*5, 3))

    preds = []
    for i, method in enumerate(args.list_names):
        pred = cv2.imread(args.list_paths[i] + name)
        pred = pred[:, -pred.shape[0]:]
        assert pred.shape[0] == pred.shape[1], "invalid shape"
        pred = cv2.resize(pred, (img_size, img_size))
        # put text on the image 
        cv2.putText(img=pred, text=method, org=(10, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 0, 255),thickness=2)
        if i % 3 != 2:
            pred = np.concatenate([pred, col_stick], axis=1)
        preds.append(pred)
        if i+1 == num_methods:
            if i % 3 == 0:
                preds.append(123*np.ones(img_size, img_size+5, 3))
                preds.append(123*np.ones(img_size, img_size, 3))
            elif i % 3 == 1:
                preds.append(123*np.ones(img_size, img_size, 3))

    ans = np.concatenate([img1, col_stick, img2, col_stick, label], axis=1)
    for row in range(row_count):
        tmp = np.concatenate(preds[(3*row):(3*(row+1))], axis=1)
        ans = np.concatenate([ans, row_stick, tmp], axis=0)

    cv2.imwrite(args.output_dir+name, ans)

    

    

    

# predictions_256_path = './output_images1/'
# predictions_divided_path = './output_images_EBDD/'
# images_path = './data_512/val/'
# output_dir = './visualized_outputs/'



# _names = [i for i in os.listdir(predictions_256_path) if not i.startswith('.')]
# _names.sort()

# for name in _names:
#     img1 = cv2.imread(images_path+'A/'+name[:-4])
#     img2 = cv2.imread(images_path+'B/'+name[:-4])
#     label_pred_divided = cv2.imread(predictions_divided_path+name[:-4])
#     label = label_pred_divided[:, :512, :]
#     pred_divided = label_pred_divided[:, -512:, :]
#     label_pred_256 = cv2.imread(predictions_256_path+name)
#     pred_256 = label_pred_256[:, -256:, :]
#     pred_512 = cv2.resize(pred_256, (512, 512))
#     upper_part = np.concatenate([np.zeros((512, 258, 3)), img1, 123*np.ones((512, 6, 3)), img2, np.zeros((512, 258, 3))], axis=1)
#     # print(label.shape, pred_512.shape, pred_divided.shape)
#     lower_part = np.concatenate([label, 123*np.ones((512, 5, 3)), pred_512, 123*np.ones((512, 5, 3)), pred_divided], axis=1)
#     # lower_part = np.concatenate([(lower_part,)*3], axis=2)
#     overall_image = np.concatenate([upper_part, lower_part], axis=0)
#     cv2.imwrite(output_dir+name, overall_image)
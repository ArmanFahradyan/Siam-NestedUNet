import torch.utils.data
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p1', "--path1", type=str, required=True, help="path of the first image")
parser.add_argument('-p2', "--path2", type=str, required=True, help="path of the second image")
parser.add_argument('-pd', "--destination_path", type=str, required=True, help="path of the destination directory")

args = parser.parse_args()

filename = args.path1.split('/')[-1]
prediction_name = "diff_map" + filename[:filename.rfind('.')]

if not os.path.exists(args.destination_path):
    os.mkdir(args.destination_path)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = './weights/snunet-32.pt'   # the path of the model
model = torch.load(path)

image1 = cv2.imread(args.path1)
image2 = cv2.imread(args.path2)

image1 = cv2.resize(image1, (256, 256))
image2 = cv2.resize(image2, (256, 256))

image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
image1 = np.moveaxis(image1, -1, 0)
image1 = image1[np.newaxis, :]
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
image2 = np.moveaxis(image2, -1, 0)
image2 = image2[np.newaxis, :]

img1 = torch.tensor(image1)
img2 = torch.tensor(image2)

model.eval()

with torch.no_grad():
    img1 = img1.float().to(dev)
    img2 = img2.float().to(dev)

    cd_preds = model(img1, img2)
    
    cd_preds = cd_preds[-1]
    _, cd_preds = torch.max(cd_preds, dim=1)

    img = cd_preds.data.cpu().numpy().squeeze() * 255
    file_path = args.destination_path + prediction_name
    cv2.imwrite(file_path + '.png', img)


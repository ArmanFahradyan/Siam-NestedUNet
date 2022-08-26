import torch.utils.data
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p1', "--path1", type=str, default='', help="path of the first image")
parser.add_argument('-p2', "--path2", type=str, default='', help="path of the second image")
parser.add_argument('-pd', "--destination_path", default='', type=str, help="path of the destination directory")

args = parser.parse_args()

path1 = args.path1
path2 = args.path2
destination_path = args.destination_path

def detect_change(path1, path2, destination_path, store_image=True, image1=None, image2=None):

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = './weights/snunet-32.pt'   # the path of the model
    model = torch.load(path)

    if image1 == None and path1 != None:
        image1 = cv2.imread(path1)
    if image2 == None and path2 != None:
        image2 = cv2.imread(path2)

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

        if store_image:
            filename = path1.split('/')[-1]
            prediction_name = "diff_map" + filename[:filename.rfind('.')]
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
            file_path = destination_path + prediction_name
            cv2.imwrite(file_path + '.png', img)
    return img

if __name__ == '__main__':
    detect_change(path1, path2, destination_path)
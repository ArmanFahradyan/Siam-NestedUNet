import argparse
import numpy as np
import os
import cv2
from predict import detect_change

parser = argparse.ArgumentParser()
parser.add_argument('-p1', "--path1", type=str, default='', help="path of the first image")
parser.add_argument('-p2', "--path2", type=str, default='', help="path of the second image")
parser.add_argument('-pd', "--destination_path", default='', type=str, help="path of the destination directory")

args = parser.parse_args()

path1 = args.path1
path2 = args.path2
destination_path = args.destination_path


def divide_and_detect(path1, path2, destination_path): 
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    assert (image1.shape == image2.shape)
    height_num = image1.shape[0] // 256
    width_num = image1.shape[1] // 256
    diff_maps = []
    for i in range(height_num):
        one_row = []
        for j in range(width_num):
            window_image1 = image1[(256*i):(256*(i+1)), (256*j):(256*(j+1)), :]
            window_image2 = image2[(256*i):(256*(i+1)), (256*j):(256*(j+1)), :]
            window_diff_map = detect_change(None, None, None, False, window_image1, window_image2)
            one_row.append(window_diff_map)
        if image1.shape[1] != 256*width_num:
            small_window_image1 = image1[(256*i):(256*(i+1)), (image1.shape[1]-256):image1.shape[1], :]
            # image1[256*i:256*(i+1), 256*width_num:image1.shape[1]]
            small_window_image2 = image2[(256*i):(256*(i+1)), (image1.shape[1]-256):image1.shape[1], :]
            # image2[256*i:256*(i+1), 256*width_num:image1.shape[1]]
            small_window_diff_map = detect_change(None, None, None, False, small_window_image1, small_window_image2)
            small_window_diff_map = small_window_diff_map[:, -(image1.shape[1]-256*width_num):]
            one_row.append(small_window_diff_map)
        diff_maps.append(one_row)
    if image1.shape[0] != 256 * height_num:
        last_row = []
        for j in range(width_num):
            small_window_image1 = image1[(image1.shape[0] - 256):image1.shape[0], (256 * j):(256 * (j + 1)), :]
            small_window_image2 = image2[(image1.shape[0] - 256):image1.shape[0], (256 * j):(256 * (j + 1)), :]
            small_window_diff_map = detect_change(None, None, None, False, small_window_image1, small_window_image2)
            small_window_diff_map = small_window_diff_map[-(image1.shape[0] - 256 * height_num):, :]
            last_row.append(small_window_diff_map)
        if image1.shape[1] != 256 * width_num:
            small_window_image1 = image1[(image1.shape[0] - 256):image1.shape[0], (image1.shape[1]-256):image1.shape[1], :]
            small_window_image2 = image2[(image1.shape[0] - 256):image1.shape[0], (image1.shape[1]-256):image1.shape[1], :]
            small_window_diff_map = detect_change(None, None, None, False, small_window_image1, small_window_image2)
            small_window_diff_map = small_window_diff_map[-(image1.shape[0] - 256 * height_num):, -(image1.shape[1] - 256 * width_num):]
            last_row.append(small_window_diff_map)
        diff_maps.append(last_row)

    # combining parts

    rows = []
    for i in range(len(diff_maps)):
        rows.append(np.concatenate(diff_maps[i], axis=1))
    result = np.concatenate(rows, axis=0)

    # saving result

    filename = path1.split('/')[-1]
    prediction_name = "diff_map" + filename[:filename.rfind('.')]
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    file_path = destination_path + prediction_name
    cv2.imwrite(file_path + '.png', result)
    return result


if __name__ == '__main__':
    divide_and_detect(path1, path2, destination_path)
    
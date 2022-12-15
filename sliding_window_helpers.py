import numpy as np
from predict import detect_change


class mylist:

    def __init__(self, l):
        self.l=l

    def __repr__(self): 
        return repr(self.l)

    def append(self, x):
        self.l.append(x)

def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def softmax(vector):
	e = np.exp(vector.l)
	return e / e.sum()

def get_weights(image_shape, kernel_size, gaussian, stride):

    weights = np.array([np.array([mylist([]) for _ in range(image_shape[1])]) for _ in range(image_shape[0])])

    flag0 = (weights.shape[0] - kernel_size) % stride != 0
    flag1 = (weights.shape[1] - kernel_size) % stride != 0

    i = 0
    while i+kernel_size <= weights.shape[0]:
        j = 0
        while j+kernel_size <= weights.shape[1]:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[i+p, j+q].append(gaussian[p, q])
            j += stride
        #---------
        if flag1:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[i+p, weights.shape[1]+q-kernel_size].append(gaussian[p, q])
        
        i += stride

    if flag0:
        j = 0
        while j+kernel_size <= weights.shape[1]:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[weights.shape[0]+p-kernel_size, j+q].append(gaussian[p, q])
            j += stride
        #---------
        if flag1:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[weights.shape[0]+p-kernel_size, weights.shape[1]+q-kernel_size].append(gaussian[p, q])
        #---------
        

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i, j]:
                weights[i, j] = list(softmax(weights[i, j]))
    return weights

def make_generator(weight_col):
    def generator():
        i = 0
        length = len(weight_col)
        while True:
            yield weight_col[i]
            i  = (i+1) % length
    return generator()

def make_generators(weights):
    result = np.empty(weights.shape, dtype=object)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = make_generator(weights[i, j])
    return result

# ----------------------------------------------

def apply_window_prediction(result, predict_window, generators, i, j):
    for p in range(predict_window.shape[0]):
        for q in range(predict_window.shape[1]):
            try:
                result[i+p, j+q] += predict_window[p, q] * next(generators[i+p, j+q])
            except:
                print(i, j, p, q)
                exit("index out of range in \"apply_window_prediction\" function")
                print()

def sliding_window_predict(img1, img2, model, kernel_size, stride, generators):
    result = np.zeros(img1.shape)
    flag0 = (img1.shape[0] - kernel_size) != 0
    flag1 = (img1.shape[1] - kernel_size) != 0
    for i in range(0, img1.shape[0]-kernel_size+1, stride):
        for j in range(0, img1.shape[1]-kernel_size+1, stride):
            window_img1 = img1[i:(i+kernel_size), j:(j+kernel_size), :]
            window_img2 = img2[i:(i+kernel_size), j:(j+kernel_size), :]
            predict_window = detect_change(path1=None, path2=None, destination_path=None, store_image=False, image1=window_img1, image2=window_img2, model=model, model_path=None)
            apply_window_prediction(result, predict_window, generators, i, j)
        if flag1:
            window_img1 = img1[i:(i+kernel_size), (img1.shape[1]-kernel_size):img1.shape[1], :]
            window_img2 = img2[i:(i+kernel_size), (img2.shape[1]-kernel_size):img2.shape[1], :]
            predict_window = detect_change(path1=None, path2=None, destination_path=None, store_image=False, image1=window_img1, image2=window_img2, model=model, model_path=None)
            apply_window_prediction(result, predict_window, generators, i, (img1.shape[1]-kernel_size))
    if flag0:
        for j in range(0, img1.shape[1]-kernel_size+1, stride):
            window_img1 = img1[(img1.shape[0]-kernel_size):img1.shape[0], j:(j+kernel_size), :]
            window_img2 = img2[(img1.shape[0]-kernel_size):img1.shape[0], j:(j+kernel_size), :]
            predict_window = detect_change(path1=None, path2=None, destination_path=None, store_image=False, image1=window_img1, image2=window_img2, model=model, model_path=None)
            apply_window_prediction(result, predict_window, generators, (img1.shape[0]-kernel_size), j)
        if flag1:
            window_img1 = img1[(img1.shape[0]-kernel_size):img1.shape[0], (img1.shape[1]-kernel_size):img1.shape[1], :]
            window_img2 = img2[(img1.shape[0]-kernel_size):img1.shape[0], (img2.shape[1]-kernel_size):img2.shape[1], :]
            predict_window = detect_change(path1=None, path2=None, destination_path=None, store_image=False, image1=window_img1, image2=window_img2, model=model, model_path=None)
            apply_window_prediction(result, predict_window, generators, (img1.shape[0]-kernel_size), (img1.shape[1]-kernel_size))
        
    return result

def compute_c_matrix(label_map, predict_map, threshold=0.5):
    tn, fp, fn, tp = 0, 0, 0, 0
    label_flatten = label_map.flatten() / 255
    predict_flatten = predict_map.flatten() / 255
    tp += predict_flatten[(label_flatten == 1) & (predict_flatten >= threshold)].sum()
    fp += predict_flatten[(label_flatten == 0) & (predict_flatten >= threshold)].sum()
    tn += (1 - predict_flatten[(label_flatten == 0) & (predict_flatten < threshold)]).sum()
    fn += (1 - predict_flatten[(label_flatten == 1) & (predict_flatten < threshold)]).sum()
    return tn, fp, fn, tp



# kernel2_size = 1024
# stride2 = 256


# def double_sliding_window_predict(img1, img2, model, kernel1_size, kernel2_size, stride1, stride2, generators1, generators2):
#     result = np.zeros(img1.shape)
#     for i in range(0, img1.shape[0]-kernel2_size+1, stride2):
#         for j in range(0, img1.shape[1]-kernel2_size+1, stride2):
#             window_img1 = img1[i:(i+kernel2_size), j:(j+kernel2_size), :]
#             window_img2 = img2[i:(i+kernel2_size), j:(j+kernel2_size), :]
#             predict_window = sliding_window_predict(window_img1, window_img2, model, kernel1_size, stride1, generators1)
#             apply_window_prediction(result, predict_window, generators2, i, j)
#     return result



# def plot_hist3d(x, y, z, dx, dy, dz):
#     plt.rcParams["figure.figsize"] = [7.50, 3.50]
#     plt.rcParams["figure.autolayout"] = True

#     fig = plt.figure()

#     ax1 = fig.add_subplot(111, projection='3d')

#     ax1.bar3d(y, x, z, dx, dy, dz, color="green")
#     ax1.axis('off')
#     plt.show()

# x = np.repeat(np.arange(image_shape[0]), image_shape[1])
# y = np.tile(np.arange(image_shape[1]), image_shape[0])
# z = np.zeros(image_shape[0]*image_shape[1])
# dx = np.ones(image_shape[0]*image_shape[1])
# dy = np.ones(image_shape[0]*image_shape[1])
# dz = []
# for s in range(weights.shape[0]):
#     for t in range(weights.shape[1]):
#         dz.append(len(weights[s, t]))
# dz = np.array(dz)

# plot_hist3d(x, y, z, dx, dy, dz)
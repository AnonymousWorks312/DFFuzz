# source: https://github.com/ARiSE-Lab/deepTest/blob/master/testgen/epoch_testgen_coverage.py

import numpy as np
import cv2
import random
def image_translation(img, params):
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

# bug: image size problem
def image_scale(img, params):
    # res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    # h = img.shape[0]
    # w = img.shape[1]
    # z = 0
    # res_shape = res.shape
    # ## check if padding is needed
    # if len(img.shape)>2:
    #     z = img.shape[2]
    # # pad_image = res.copy()
    # pad_h = res_shape[0] - h
    # pad_w = res_shape[1] - w
    # # print(pad_h,pad_w)
    # if pad_h < 0 or pad_w < 0:
    #     total_h = abs(pad_h)
    #     total_w = abs(pad_w)
    #     x = random.randint(0, total_h)
    #     y = random.randint(0, total_w)
    #     res = cv2.copyMakeBorder(res, x, total_h-x,y, total_w-y,
    #                                    cv2.BORDER_CONSTANT,
    #                                    value=(0))
    # ## check if crop is needed
    # if pad_h > 0 or pad_w > 0:
    #     x = random.randint(0, res_shape[1] - w)
    #     y = random.randint(0, res_shape[0] - h)
    #     res = res[y:y + h, x:x + w]
    # print(res.shape)

    rows, cols, ch = img.shape
    if len(img.shape) > 2:
        ch = img.shape[2]

    res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    res = res.reshape((res.shape[0], res.shape[1], ch))
    y, x, z = res.shape

    if params > 1:  # need to crop
        startx = x // 2 - cols // 2
        starty = y // 2 - rows // 2
        return res[starty:starty + rows, startx:startx + cols]
    elif params < 1:  # need to pad
        sty = (rows - y) // 2
        stx = (cols - x) // 2
        #     res = cv2.copyMakeBorder(res, x, total_h-x,y, total_w-y,
        #                                    cv2.BORDER_CONSTANT,
        #                                    value=(0))
        return np.pad(res, pad_width=((sty, rows - y - sty), (stx, cols - x - stx),(0,0)), mode='constant',
                      constant_values=0)
    return res


# bug: image size problem
def image_shear(img, params):
    rows, cols, ch = img.shape
    # print('params',params)
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows),flags=cv2.INTER_AREA)
    return dst

def image_noise(img,params):
    if params == 1:  # Gaussian-distributed additive noise.
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return noisy.astype(np.uint8)
    elif params == 2:  # Replaces random pixels with 0 or 1.
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        return out
    elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = img + img * gauss
        return noisy.astype(np.uint8)

def image_contrast(img, params):
    alpha = params
    # print('params', params)
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)                                  # new_img = img + beta
    return new_img

def image_blur(img, params):
    img_type = img.dtype
    if(np.issubdtype(img_type, np.integer)):
        img = np.uint8(img)
    else:
        img = np.float32(img)
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    #if params == 9:
    #    blur = cv2.blur(img, (6, 6))
    if params == 9:
    #     blur = cv2.bilateralFilter(img, 9, 75, 75)
        blur = cv2.bilateralFilter(img, 6, 50, 50)
    blur = blur.astype(img_type)
    return blur

def image_erode(img,params):
    # params kernel (5,5)
    kernel = np.ones(params, np.uint8)
    new_image = cv2.erode(img, kernel, iterations=1)
    return new_image

def image_dilate(img,params):
    # params (5,5)
    kernel = np.ones(params, np.uint8)
    new_image = cv2.dilate(img, kernel, iterations=1)
    return new_image

def reverse_color_patch(image,params):
    i = params[0]
    j = params[1]
    # params i from 0 to height-2 j ditto
    image = np.array(image, dtype=float)
    part = image[i:2+i,j:2+j].copy()
    reversed_part = 255-part
    image[i:2+i,j:2+j] = reversed_part
    return image

def shuffle_patch(image,params):
    # params i from 0 to height-2 j ditto
    i = params[0]
    j = params[1]
    image = np.array(image, dtype=float)
    part = image[i:2+i,j:2+j].copy()
    part_r = part.reshape(-1,1)
    np.random.shuffle(part_r)
    part_r = part_r.reshape(part.shape)
    image[i:2+i,j:2+j] = part_r
    return image


def white_patch(image,params):
  i = params[0]
  j = params[1]
  image = np.array(image, dtype=float)
  image[i:2+i,j:2+j]=255
  return image


def black_patch(image,params):
  i = params[0]
  j = params[1]
  image = np.array(image, dtype=float)
  image[i:2+i,j:2+j]=0
  return image


def reverse_pixel(image,params):
    # params i from 0 to height j ditto
    i = params[0]
    j = params[1]
    image = np.array(image, dtype=float)
    image = np.array(image, dtype=float)
    part = image[i:i+1,j:j+1].copy()
    reversed_part = 255-part
    image[i:i+1,j:j+1] = reversed_part
    return image

def white_pixel(image,params):
    i = params[0]
    j = params[1]
    image = np.array(image, dtype=float)
    image[i:i+1,j:j+1]=255
    return image

def black_pixel(image,params):
    i = params[0]
    j = params[1]
    image = np.array(image, dtype=float)
    image[i:i+1,j:j+1]=0
    return image

def switch_pixel(image,params):
    i_1 = params[0][0]
    j_1 = params[0][1]
    i_2 = params[1][0]
    j_2 = params[1][1]
    image = np.array(image, dtype=float)
    part_1 = image[i_1:i_1+1,j_1:j_1+1].copy()
    part_2 = image[i_2:i_2+1,j_2:j_2+1].copy()
    image[i_1:i_1+1,j_1:j_1+1] = part_2
    image[i_2:i_2+1,j_2:j_2+1] = part_1
    return image

def gaussian_noise(img,params):
        row, col, ch = img.shape
        mean = 0
        var = params
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return noisy.astype(np.uint8)

def saltpepper_noise(img,params):
 # Replaces random pixels with 0 or 1.
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        return out

def multiplicative_noise(img,params):
        # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        row, col, ch = img.shape
        mean = 0
        var = params
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + img * gauss
        return noisy.astype(np.uint8)

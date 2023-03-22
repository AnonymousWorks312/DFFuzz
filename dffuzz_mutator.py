import utils.image_transforms as Image_transforms
from params.parameters import Parameters
import random
# random.seed(2021)
deepchange = Parameters()

deepchange.K = 64
deepchange.batch1 = 1
deepchange.batch2 = 1
deepchange.p_min = 0.0 # 0.3
deepchange.gamma = 5
deepchange.alpha = 0.02
deepchange.beta = 0.20
deepchange.TRY_NUM = 50

def image_translation(img):
    parameters_list = [(i,j) for i in range(-3,3) for j in range(-3,3)]
    return Image_transforms.image_translation(img,random.choice(parameters_list))

def image_scale(img):
    # parameters_list = random.uniform(0.7,1.2)#[1.05]#[i * 0.1 for i in range(7, 12)]
    return Image_transforms.image_scale(img,random.uniform(0.7,1.2))

def image_shear(img):
    # parameters_list = [0.1 * k for k in range(-6,6)]
    return Image_transforms.image_shear(img,random.uniform(-0.6,0.6))

def image_rotation(img):
    #parameters_list = list(range(-50, 50)) # random.uniform(-50, 50)
    return Image_transforms.image_rotation(img,random.uniform(-50, 50))

def image_contrast(img):
    #parameters_list = [i*0.1 for i in range(5, 15)]
    return Image_transforms.image_contrast(img,random.uniform(0.5, 1.5))

def image_brightness(img):
    #parameters_list = list(range(-20, 20))
    return Image_transforms.image_brightness(img,random.uniform(-20, 20))

def image_blur(img):
    parameters_list = [k + 1 for k in range(9)]
    return Image_transforms.image_blur(img,random.choice(parameters_list))

def image_noise(img):
    parameters_list = [1,2,3]
    return Image_transforms.image_noise(img,random.choice(parameters_list))

# from OpenCV
def image_erode(img):
    parameters_list = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,3)]
    return Image_transforms.image_erode(img,random.choice(parameters_list))

def image_dilate(img):
    parameters_list = [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(3,3)]
    return Image_transforms.image_dilate(img,random.choice(parameters_list))

# patch_wise inherited from PRIMA
def image_reverse_patch(img):
    # print(img.shape)
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
    return Image_transforms.reverse_color_patch(img,random.choice(parameters_list))

def image_shuffle_patch(img):
    # print(img.shape)
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
    return Image_transforms.shuffle_patch(img,random.choice(parameters_list))

def image_white_patch(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
    return Image_transforms.white_patch(img,random.choice(parameters_list))

def image_black_patch(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
    return Image_transforms.black_patch(img,random.choice(parameters_list))

# pixel_wise inspired by one_pixel_attack
def image_black_pixel(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.black_pixel(img,random.choice(parameters_list))

def image_white_pixel(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.white_pixel(img,random.choice(parameters_list))

def image_reverse_pixel(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.reverse_pixel(img,random.choice(parameters_list))

def image_switch_pixel(img):
    img_w,img_d = img.shape[0],img.shape[1]
    parameters_list = [(i,j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.switch_pixel(img,random.sample(parameters_list,2))

def gaussian_noise(img):
    # params var:
    parameters_list = [2,1,0.5,0.1,0.05,0.01,0.005,0.001]
    return Image_transforms.gaussian_noise(img,random.choice(parameters_list))

def sp_noise(img):
    # params salt and pepper has many positions, no need to set any parameters:
    parameters_list = []
    return Image_transforms.saltpepper_noise(img,[])

def multiplicative_noise(img):
    # params var:
    parameters_list = [1,0.5,0.1,0.05,0.01,0.005,0.001]
    return Image_transforms.multiplicative_noise(img,random.choice(parameters_list))

# import numpy as np
# test_images = np.load('../mnist_sample_2000.npz')['x']
# test_labels = np.load('../mnist_sample_2000.npz')['y']
# test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
# from PIL import Image
#
# mutated_img_1 = test_images[0]
# mutated_img_2 = image_rotation(test_images[0])
#
# print(np.sum(mutated_img_1-mutated_img_2!=0))
# import matplotlib.pyplot as plt
# plt.subplot(1, 2, 1)
# plt.imshow(mutated_img_1)
# plt.subplot(1, 2, 2)
# plt.imshow(mutated_img_2)
#
# # plt.imshow(mutated_img_1,mutated_img_2) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()
# def image_reverse_pixel(img):
#     # print(img.shape)
#     img_w,img_d = img.shape[0],img.shape[1]
#     parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
#     return Image_transforms.reverse_color_patch()
#
# def image_white_pixel(img):
#     img_w,img_d = img.shape[0],img.shape[1]
#     parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
#     return Image_transforms.image_dilate(img,random.choice(parameters_list))
#
# def image_black_pixel(img):
#     img_w,img_d = img.shape[0],img.shape[1]
#     parameters_list = [(i,j) for i in range(img_w-2) for j in range(img_d-2)]
#     return Image_transforms.image_dilate(img,random.choice(parameters_list))

def get_mutation_ops_name():
    # return ['translation', 'scale', 'shear', 'rotation', 'contrast', 'brightness', 'blur','erode','dilate',
    #         'reverse_patch','white_patch','black_patch','shuffle_patch','black_pixel','white_pixel',
    #         'switch_pixel','reverse_pixel','gauss_noise','multiplicative_noise','saltpepper_noise']

    return ['translation', 'scale', 'shear', 'rotation', 'contrast', 'brightness', 'blur','erode','dilate',
            'reverse_patch','white_patch','black_patch','gauss_noise','multiplicative_noise','saltpepper_noise']

def get_mutation_func(name):
    if name == 'translation':
        return image_translation
    elif name == 'scale':
        return image_scale
    elif name == 'shear':
        return image_shear
    elif name == 'rotation':
        return image_rotation
    elif name == 'contrast':
        return image_contrast
    elif name == 'brightness':
        return image_brightness
    elif name == 'blur':
        return image_blur
    elif name == 'noise':
        return image_noise
    elif name == 'erode':
        return image_erode
    elif name == 'dilate':
        return image_dilate
    elif name == 'reverse_patch':
        return image_reverse_patch
    elif name == 'black_patch':
        return image_black_patch
    elif name == 'white_patch':
        return image_white_patch
    elif name == 'shuffle_patch':
        return image_shuffle_patch
    elif name == 'black_pixel':
        return image_black_pixel
    elif name == 'white_pixel':
        return image_white_pixel
    elif name == 'black_pixel':
        return image_black_pixel
    elif name == 'reverse_pixel':
        return image_reverse_pixel
    elif name == 'switch_pixel':
        return image_switch_pixel
    elif name == 'gauss_noise':
        return gaussian_noise
    elif name == 'multiplicative_noise':
        return multiplicative_noise
    elif name == 'saltpepper_noise':
        return sp_noise
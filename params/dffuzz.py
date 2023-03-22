import numpy as np
import itertools
import utils.image_transforms as image_transforms
from params.parameters import Parameters

deepchange = Parameters()

deepchange.K = 64
deepchange.batch1 = 1
deepchange.batch2 = 1
deepchange.p_min = 0.0 # 0.3
deepchange.gamma = 5
deepchange.alpha = 0.02
deepchange.beta = 0.20
deepchange.TRY_NUM = 50
deepchange.MIN_FAILURE_SCORE = -100
deepchange.framework_name = 'dffuzz'
# translation = list(itertools.product([getattr(image_transforms,"image_translation")], [(10+10*k,10+10*k) for k in range(10)]))
# scale = list(itertools.product([getattr(image_transforms, "image_scale")], [(1.5+0.5*k,1.5+0.5*k) for k in range(10)]))
# shear = list(itertools.product([getattr(image_transforms, "image_shear")], [(-1.0+0.1*k,0) for k in range(10)]))
# rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], [3+3*k for k in range(10)]))
# contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [1.2+0.2*k for k in range(10)]))
# brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10+10*k for k in range(10)]))
# blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k+1 for k in range(10)]))




translation = list(itertools.product([getattr(image_transforms, "image_translation")],list(range(-3, 3)) ))
scale = list(itertools.product([getattr(image_transforms, "image_scale")], [i*0.1 for i in range(7, 12)]))
shear = list(itertools.product([getattr(image_transforms, "image_shear")], [0.1*k for k in range(-6,6)]))
rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-50, 50))))
contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [i*0.1 for i in range(5, 13)]))
brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], list(range(-20, 20))))
blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(9)]))
noise = list(itertools.product([getattr(image_transforms, "image_noise")],[1,2,3]))
# rotation = list(
#     itertools.product([getattr(image_transforms, "image_rotation")], [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]))
# contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [1.2 + 0.2 * k for k in range(10)]))
# brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(10)]))
# blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))
# shear = list(itertools.product([getattr(image_transforms, "image_shear")], [-0.5+0.1*k for k in range(10)]))
# scale = list(itertools.product([getattr(image_transforms, "image_scale")], [(1+0.05*k,1+0.05*k) for k in [-5,-4,-3,-2,-1,1,2,3,4,5]]))
# noise = list(itertools.product([getattr(image_transforms, "image_noise")], [0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1]))

deepchange.G = translation + rotation + scale + shear
deepchange.P = contrast + brightness + blur + noise
print(deepchange.G+deepchange.P)

# deepchange.G = rotation + noise
# deepchange.P = contrast + brightness + blur

deepchange.save_batch = False

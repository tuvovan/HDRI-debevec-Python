import os
import glob
import cv2
import argparse
import numpy as np 
from utils import *

def main(config):
    list_file, exps = load(config.test_path)
    stack = read(list_file)

    num_channels = stack.shape[-1]
    hdr_img = np.zeros(stack[0].shape, dtype=np.float64)

    for c in range(num_channels):
        layer_stack = [img[:,:,c] for img in stack]
        
        sample = sample_intensity(layer_stack)

        curve = estimate_curve(sample, exps, 100.)

        img_rad = computeRadiance(np.array(layer_stack), exps, curve)

        hdr_img[:,:, c] = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if config.tonemap =='gamma':
        output = np.uint8(globalTonemap(hdr_img, 1.3) * 255.)
    else:
        tm = cv2.createTonemapMantiuk()
        output = np.uint8(255. * tm.process((hdr_img/255.).astype(np.float32)))

    if config.cmap:
        from matplotlib.pylab import cm
        colorize = cm.jet
        cmap = np.float32(cv2.cvtColor(np.uint8(hdr_img), cv2.COLOR_BGR2GRAY)/255.)
        cmap = colorize(cmap)
        cv2.imwrite(os.path.join(config.test_path, 'cmap.jpg'), np.uint8(cmap*255.))

    template = stack[len(stack)//2]
    image_tuned = intensityAdjustment(output, template)
    output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output.astype(np.uint8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default="imgs/night01/")
    parser.add_argument('--tonemap', type=str, default=' ')
    parser.add_argument('--cmap', type=bool, default=False)

    config = parser.parse_args()
    out = main(config)

    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), out)
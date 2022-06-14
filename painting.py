import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import glob

from baseline.DRL.actor import *
from baseline.Renderer.stroke_gen import *
from baseline.Renderer.model import *

def run_painting(path_actor, path_renderer, img, max_step, divide, imgid =0):

    def decode(x, canvas): # b * (10 + 3)
        x = x.view(-1, 10 + 3)
        stroke = 1 - Decoder(x[:, :10])
        stroke = stroke.view(-1, width, width, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, 5, 1, width, width)
        color_stroke = color_stroke.view(-1, 5, 3, width, width)
        res = []
        for i in range(5):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(canvas)
        return canvas, res

    def small2large(x):
        # (d * d, width, width) -> (d * width, d * width)    
        x = x.reshape(divide, divide, width, width, -1)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(divide * width, divide * width, -1)
        return x

    def large2small(x):
        # (d * width, d * width) -> (d * d, width, width)
        x = x.reshape(divide, width, divide, width, 3)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(canvas_cnt, width, width, 3)
        return x

    def smooth(img):
        def smooth_pix(img, tx, ty):
            if tx == divide * width - 1 or ty == divide * width - 1 or tx == 0 or ty == 0: 
                return img
            img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
            return img

        for p in range(divide):
            for q in range(divide):
                x = p * width
                y = q * width
                for k in range(width):
                    img = smooth_pix(img, x + k, y + width - 1)
                    if q != divide - 1:
                        img = smooth_pix(img, x + k, y + width)
                for k in range(width):
                    img = smooth_pix(img, x + width - 1, y + k)
                    if p != divide - 1:
                        img = smooth_pix(img, x + width, y + k)
        return img

    def save_img(res, imgid, divide=False):
        output = res.detach().cpu().numpy() # d * d, 3, width, width    
        output = np.transpose(output, (0, 2, 3, 1))
        if divide:
            output = small2large(output)
            output = smooth(output)
        else:
            output = output[0]
        output = (output * 255).astype('uint8')
        output = cv2.resize(output, origin_shape)
        cv2.imwrite('output/generated' + str(imgid) + '.png', output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = 128
    canvas_cnt = divide * divide
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    # img = np.array(pil_image)
    # if len(img.shape) == 3:
    #     img = img[:, :, ::-1].copy()
    origin_shape = (img.shape[1], img.shape[0])

    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width - 1.)
            coord[0, 1, i, j] = j / (width - 1.)
    coord = coord.to(device)

    Decoder = FCN()
    Decoder.load_state_dict(torch.load(path_renderer))

    actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(path_actor))
    actor = actor.to(device).eval()
    Decoder = Decoder.to(device).eval()

    canvas = torch.zeros([1, 3, width, width]).to(device)

    patch_img = cv2.resize(img, (width * divide, width * divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.

    pathlib.Path("output").mkdir(parents=True, exist_ok=True)
    files = glob.glob('output')
    for f in files:
        os.system(f"del {f} /Q")

    with torch.no_grad():
        if divide != 1:
            max_step = max_step // 2
        for i in range(max_step):
            stepnum = T * i / max_step
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            canvas, res = decode(actions, canvas)
            print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
            for j in range(5):
                save_img(res[j], imgid)
                imgid += 1
        if divide != 1:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))    
            canvas = cv2.resize(canvas, (width * divide, width * divide))
            canvas = large2small(canvas)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)
            T = T.expand(canvas_cnt, 1, width, width)
            for i in range(max_step):
                stepnum = T * i / max_step
                actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                canvas, res = decode(actions, canvas)
                print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                for j in range(5):
                    save_img(res[j], imgid, True)
                    imgid += 1

    return True
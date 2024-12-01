from tkinter import TRUE
import torch, cv2, time, os
from PIL import Image
import os.path as osp
import numpy as np
import torch.nn.functional as F
from utils_Saliency  import  SalEval, Logger
from Models_saliency.SAMNet import FastSal as net
import random as rd
from  torchvision import utils as vutils
#import gol as gl

def get_saliencyMaps(images_batch,batch_size):
    width =images_batch.size(2)
    height =images_batch.size(2)
    savedir = './Outputs'
    gpu = TRUE
    pretrained = './Pretrained/SAMNet_with_ImageNet_pretrain.pth'

    if not osp.isdir(savedir):
        os.mkdir(savedir)

    model = net()
    state_dict = torch.load(pretrained,map_location=torch.device('cpu'))
    if list(state_dict.keys())[0][:7] == 'module.':
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print('Model resumed from %s' % pretrained)

    if gpu:
        model = model.cuda()
    model.eval()
    mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
    std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
    
    image_size = images_batch.size(2)
    image = torch.zeros(batch_size,3,image_size,image_size)
    for k in range(batch_size):
        image_one = images_batch[k].cpu().numpy()
        image_one = image_one *255
        image_one = image_one.astype(np.uint8)
        image_one = np.transpose(image_one,(2,1,0))
        image_one = np.transpose(image_one,(1,0,2))    #矩阵转置        
        img = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
        cv2.imwrite(osp.join(savedir, '000200.jpg'),image_one)        

        image_one = Image.fromarray(img)
        #image_one.show()
        image_one = image_one.convert('RGB')  
        ######################################
        image_one = np.array(image_one, dtype=np.float32)
        image_one = (image_one - mean) / std
        image_one = cv2.resize(image_one, (width, height), interpolation=cv2.INTER_LINEAR)
        image_one = image_one.transpose((2, 0, 1))
        image_one = torch.from_numpy(image_one).unsqueeze(0)
        image[k,:]=image_one
    if gpu:
        image = image.cuda()
    with torch.no_grad():
        pred = model(image)[:, 0, :, :].unsqueeze(1)

    assert pred.shape[-2:] == image.shape[-2:], '%s vs. %s' % (str(pred.shape), str(image.shape))
    pred = F.interpolate(pred, size=[height, width], mode='bilinear', align_corners=False)             #线性插值
    pred = pred.squeeze(1)
    
    sali_image = torch.zeros(batch_size,image_size,image_size)
    #pix_sali_index_num = torch.zeros(batch_size,1)    

    for m in range(batch_size):
        pred_one = (pred[m] * 255).cpu().numpy().astype(np.uint8)

        ix_one=np.array(np.where(pred_one>20))
        #####################
        ix_image = np.zeros((height,width))
        for i in range(len(ix_one[0])):
            ix_image[ix_one[0,i],ix_one[1,i]] = 255
        #serial = gl.get_value('serial') * batch_size+ m
    
        prub_name = 'saliency/%s.jpg' % (m)
        cv2.imwrite(osp.join(savedir, prub_name), ix_image)
        ix_image = ix_image / 255
        sali_image[m,:,:] = torch.from_numpy(ix_image)
        ##############################
    return (sali_image)


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(rd.randint(start, stop))
    return random_list
import random
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as tf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import GazeNet

device = 'cuda'

class MPIIFaceGaze:
    def __init__(self, subjects, img_dir, img_size=(256, 256)):
        self.img_dir = Path(img_dir).expanduser().resolve()
        self.img_size = img_size
        self.anns = []
        for subject_id in subjects:
            with open(self.img_dir.parent / f'{subject_id:02d}.json') as f:
                self.anns.extend(json.load(f))
       
    def __len__(self):
        return len(self.anns)
 
    def __getitem__(self, idx):
        ann = self.anns[idx]
        img = Image.open(self.img_dir / ann['image'])
        img = img.resize(self.img_size)
        img = tf.to_tensor(img)
        g = torch.FloatTensor(ann['g'])
        h = torch.FloatTensor(ann['h'])
        return img, g, h


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 0]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi

def train(epoch, model, optimizer, criterion, train_loader):
    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()

    start = time.time()

    for image, gaze, headpose in iter(train_loader):
        image = image.to(device)
        gaze = gaze.to(device)

        optimizer.zero_grad()
        predict = model(image)
        loss = criterion(predict, gaze)
        loss.backward()
        optimizer.step()

        angle_error = compute_angle_error(predict, gaze).mean()

        num = image.size(0)

        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        #pbar.set_postfix(loss=loss_meter.val, angle_error=angle_error_meter.val)
        #pbar.update(num)
    
    #pbar.set_postfix(loss=loss_meter.avg, angle_error=angle_error_meter.avg)
    elapse = time.time()-start
    print(f'Running time={elapse:3.3f}')
    print(f'Train: Loss={loss_meter.avg:3.4f}, angle_error={angle_error_meter.avg:3.4f}')

def test(epoch, model, criterion, test_loader):
    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()

    for image, gaze, headpose in iter(test_loader):
        image = image.to(device)
        gaze = gaze.to(device)

        predict = model(image)
        loss = criterion(predict, gaze)

        angle_error = compute_angle_error(predict, gaze).mean()

        num = image.size(0)

        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        #pbar.set_postfix(loss=loss_meter.val, angle_error=angle_error_meter.val)
        #pbar.update(num)

    #pbar.set_postfix(loss=loss_meter.avg, angle_error=angle_error_meter.avg)
    print(f'Test: Loss={loss_meter.avg:3.4f}, angle_error={angle_error_meter.avg:3.4f}')

    return loss_meter.avg

def main():
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # reproducibility
    torch.backends.cudnn.deterministic = True

    # data directory
    data_dir = Path('./mpii_data/img/').expanduser().resolve()
    # output directory
    out_dir = Path('./output/')
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)

    # data loaders
    train_set = MPIIFaceGaze(range(0,14), data_dir)
    test_set = MPIIFaceGaze([14], data_dir)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=2)

    # model
    model = GazeNet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # training epochs
    for epoch in range(20):
        print('Epoch ', epoch, flush=True)

        #with tqdm(total=len(train_set), desc='Train') as pbar:
        train(epoch, model, optimizer, criterion, train_loader)
        
        with torch.no_grad():
            #with tqdm(total=len(test_set), desc='Test') as pbar:
            test_loss = test(epoch, model, criterion, test_loader)
        
        scheduler.step(test_loss)

        torch.save(model.state_dict(), str(epoch) + '_model_state.pth')

if __name__ == '__main__':
    main()
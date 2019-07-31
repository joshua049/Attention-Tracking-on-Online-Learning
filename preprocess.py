import json
import h5py
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
 
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf
 
 
def preprocess(mat_dir, dst_dir):
    mat_dir = Path(mat_dir).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()
    img_dir = dst_dir / 'img'
    for dir_ in [dst_dir, img_dir]:
        if img_dir.exists():
            shutil.rmtree(str(dir_))
        dir_.mkdir(parents=True)
 
    mat_paths = sorted(list(mat_dir.glob('*.mat')))
    mat_paths = tqdm(mat_paths)
 
    for subject_id, path in enumerate(mat_paths):
        anns = []
 
        with h5py.File(path) as f:
            imgs = np.uint8(f['Data']['data'])
            lbls = np.array(f['Data']['label'])
        imgs = np.transpose(imgs, [0, 2, 3, 1]) # [N, H, W, C]
        imgs = imgs[..., ::-1] # BGR -> RGB
        lbls = lbls.tolist() # [N, 16]
 
        for img, lbl in zip(imgs, lbls):
            idx = subject_id * 3000 + len(anns)
            dst = img_dir / f'{idx:08d}.jpg'
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img.save(dst)
 
            anns.append({
                'image': dst.name,
                'g': lbl[0:2],
                'h': lbl[2:4],
                'landmarks': lbl[4:]
            })
       
        with (dst_dir / f'{subject_id:02d}.json').open('w') as f:
            json.dump(anns, f, indent=2, ensure_ascii=False)
 
 
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
 
 
if __name__ == '__main__':
    preprocess('~/dataset/MPIIFaceGaze_normalizad/', './mpii_data/')
   
   
    # data = MPIIFaceGaze([0], './mpii_data/img/', [256, 256])
    # print(len(data))
 
    # img, g, h = data[-333]
    # img = tf.to_pil_image(img)
    # g = g.numpy() # pitch, yawㄗㄣ
    # h = h.numpy() # pitch, yaw
 
    # print(g)
    # # print(h)
 
    # import util
    # import matplotlib.pyplot as plt
 
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # util.draw_arrow(ax, g, color='r')
    # util.draw_arrow(ax, h, color='b')
    # plt.show()
 
    # from mpl_toolkits.mplot3d import Axes3D
 
    # fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    # ax.view_init(10, 10)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_xlim(-1, +1)
    # ax.set_ylim(-1, +1)
    # ax.set_zlim(-1, +1)
    # ax.set_aspect('equal')
    # util.draw_shpere(ax)
    # util.draw_arrow3d(ax, g)
 
    # plt.show()

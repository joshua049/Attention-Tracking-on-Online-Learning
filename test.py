import numpy as np
import torch
from models import GazeNet as Model
from torchvision.transforms import functional as tf

device = 'cuda'
model = Model().to(device)
#state = torch.load('model_state.pth')
#model.load_state_dict(state['state_dict'], strict=False)
model.load_state_dict(torch.load('model_state.pth'))
model.eval()

def run(image):
    tmp = np.zeros((1, 2))
    tmp = tf.to_tensor(tmp)
    tmp = tmp.to(device)
    
    image = tf.to_tensor(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        angle = model(image)
    
    angle = angle.to('cpu')
    angle = angle.squeeze(0)

    return angle




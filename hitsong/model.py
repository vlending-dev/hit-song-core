import cv2
import numpy as np
import torch
import torch.nn as nn

from hitsong.file import Audio


device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")


def conv_layer(ni, nf, ks=3, stride=1, act=True):
    bn = nn.BatchNorm2d(nf)
    layers = [nn.Conv2d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False), bn]
    act_fn = nn.ReLU(inplace=True)
    if act:
        layers.append(act_fn)
    return nn.Sequential(*layers)


def conv_layer_averpl(ni, nf):
    aver_pl = nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.Sequential(conv_layer(ni, nf), aver_pl)


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Model:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            conv_layer_averpl(1, 64),
            ResBlock(64),
            conv_layer_averpl(64, 64),
            ResBlock(64),
            conv_layer_averpl(64, 128),
            ResBlock(128),
            conv_layer_averpl(128, 256),
            ResBlock(256),
            conv_layer_averpl(256, 512),
            ResBlock(512),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(2048, num_classes * 5),
            nn.Linear(num_classes * 5, num_classes)
        )


class ModelLoader:
    def __init__(self, device: str = None, base_path: str = None, model_path: str = 'model.pt', num_classes: int = 0):
        self.device = device_cuda if device == 'cuda' else device_cpu
        self.path = base_path
        self.model_path = model_path
        self.num_classes = 0
        self.model = None
        self.storage = None
        self.load_model(num_classes=num_classes)

    def load_model(self, num_classes):
        self.num_classes = num_classes
        base_model = Model(num_classes=self.num_classes)
        base_model_val = base_model.model.to(self.device)

        # Load model
        base_model_val.load_state_dict(torch.load(self.model_path, map_location=self.device))
        base_model_val.eval()

        # Discard last Softmax Layer
        removed_model = list(base_model.model.children())[:-1]
        self.model = nn.Sequential(*removed_model)

    @torch.no_grad()
    def calc_prediction(self, audio: Audio):
        if not audio.exists_audio_file():
            return None
        files = audio.slice_files()
        if not files:
            return None
        matrix_size = self.num_classes * 5
        prediction = torch.zeros((1, matrix_size)).to(self.device)
        images = []
        for f in files:
            image = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)

        images = np.array(images)
        images = images[:, None, :, :]
        images = images / 255.
        for image in images:
            image = image[None, :, :, :]
            image_trans = torch.from_numpy(image.astype(np.float32)).to(self.device)
            prediction = prediction + self.model(image_trans)
        prediction /= len(files)
        return prediction.tolist()

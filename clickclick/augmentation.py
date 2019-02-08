import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms

# prior
mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]

CROP_SIZE = 224
MANIP_PROB = 0.5 # pavel's idea


class CameraAugmentor:
    def __init__(self, train_mode=True, quality_ls=None, bicubic_ls=None, gamma_ls=None):
        self.train_mode = train_mode
        
        self.manip_ls = []
        
        self.quality_ls = [70, 90] if quality_ls is None else quality_ls
        self.bicubic_ls = [0.5, 0.8, 1.5, 2.0] if bicubic_ls is None else bicubic_ls
        self.gamma_ls = [0.8, 1.2] if gamma_ls is None else bicubic_ls

        for quality in self.quality_ls:
            self.manip_ls.append(lambda img: self.jpg_manip(img, quality))

        for bicubic in self.bicubic_ls:
            self.manip_ls.append(lambda img: self.bicubic_manip(img, bicubic))

        for gamma in self.gamma_ls:
            self.manip_ls.append(lambda img: self.gamma_manip(img, gamma))
            
        self.transform = CameraAugmentor.get_transforms()
    

    @staticmethod
    def get_transforms():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=sigma)
        ])

    
    @staticmethod
    def jpg_manip(image, quality):
        buffer = BytesIO()
        img = Image.fromarray(image)
        img.save(buffer, format='jpeg', quality=quality)
        buffer.seek(0)
        img_jpg = Image.open(buffer)
        img_jpg = np.array(img_jpg)
        return img_jpg


    @staticmethod
    def gamma_manip(image, gamma):
        img_gamma = np.uint8(cv2.pow(image / 255., gamma) * 255.)
        return img_gamma

        
    @staticmethod
    def bicubic_manip(image, scale):
        img_bicubic = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img_bicubic
    
 
    def __call__(self, image, aug_type):
        if aug_type in range(8):
            manip_fnc = self.manip_ls[aug_type]
            img = manip_fnc(image)
        elif aug_type == 8:
            img = np.rot90(image, 1)
        elif aug_type == 9:
            img = np.rot90(image, 2)
        else:
            img = np.rot90(image, 3)
        
        # use additional augmentation with probability `MANIP_PROB`
        # from PavelOstyakov
        if self.train_mode and np.random.random() < MANIP_PROB:
            manip_fnc = np.random.choice(self.manip_ls)
            img = manip_fnc(img)
        
        img = self.transform(img)
        return img                
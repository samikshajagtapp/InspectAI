import cv2
from pathlib import Path
from torchvision import transforms
import albumentations as A

img_path = list(Path('mvtec_dataset/bottle/test/good').glob('*.png'))[0]
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Albumentations
a_transform = A.Compose([A.Resize(256, 256, always_apply=True)])
a_img = a_transform(image=img_rgb)['image']

# Torchvision
t_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256))])
t_img = np.array(t_transform(img_rgb))

diff = np.abs(a_img.astype(float) - t_img.astype(float))
print('Max diff:', diff.max(), 'Mean diff:', diff.mean())

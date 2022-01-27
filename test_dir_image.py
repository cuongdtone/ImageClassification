from utils.load_model import Model
import cv2
from glob import glob
from utils.plot import plot_image

path_image = 'dataset/train/thekhac'

list_images = glob(path_image + '/*.jpg') + glob(path_image + '/*.jpeg') + glob(path_image + '/*.png')
list_images.sort()
print("Number image: ", len(list_images))
#Load model
model = Model("/home/cuong/Desktop/bianry_classification/runs/exp0", device='cpu')
for image in list_images:
    image = cv2.imread(image)
    pred = model.predict(image)
    print(pred)
    plot_image(image, pred[0] +': %d%%'%(pred[1]))

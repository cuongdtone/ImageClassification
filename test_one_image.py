from utils.load_model import Model
from utils.plot import plot_image
import cv2

model = Model("/home/cuong/Desktop/bianry_classification/runs/exp0", device='cpu')
image = cv2.imread("dataset_example/thehdv/thehdvndc.ND-Cu-1.jpg")
pred = model.predict(image)
print(pred)

plot_image(image, pred[0] +': %d%%'%(pred[1]))




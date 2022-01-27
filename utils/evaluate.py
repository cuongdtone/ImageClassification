from glob import glob
import cv2
from utils.load_model import Model
from sklearn import metrics
from utils.plot import plot_cm


def evaluate_cm(model, test_set_path, save_cm='', normalize = True, show=True):
    truth_label = []
    pred_label = []
    for cls in model.class_name:
        list_img = glob(test_set_path + '/' + cls + '/*.jpg')
        for i in list_img:
            image = cv2.imread(i)
            pred = model.predict(image)
            truth_label.append(model.class_name_id[cls])
            pred_label.append(model.class_name_id[pred[0]])
    #print(truth_label)
    #print(pred_label)
    CM = metrics.confusion_matrix(truth_label, pred_label)
    plot_cm(CM, save_dir=save_cm, names=model.class_name, normalize=normalize, show=show)


from utils.evaluate import evaluate_cm
from utils.load_model import Model

test_set_path = "dataset/test"
model = Model("/home/cuong/Desktop/bianry_classification/runs/exp1")
evaluate_cm(model, test_set_path, normalize=False)
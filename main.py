import shap
from src.preprocess import Data
from src.model import Classifier

if __name__ == '__main__':
    data = Data(csv_path='data/prostate_cancer.csv')
    classifier = Classifier(x_train=data.x_train, x_test=data.x_test, y_train=data.y_train, y_test=data.y_test)

    explainer = shap.Explainer(classifier.model, data.data.drop(['diagnosis_result'], axis=1))
    shap_values = explainer(data.data.drop(['diagnosis_result'], axis=1))
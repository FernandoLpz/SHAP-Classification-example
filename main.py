import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    x, y = load_breast_cancer(return_X_y=True, as_frame=False)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    explainer = shap.TreeExplainer(rf)

    shap_values = explainer(x_train)


    # shap.initjs()   

    # shap.plots.force(shap_values[0])
    # shap.force_plot(explainer.expected_value[0], shap_values[0])
    shap.plots.waterfall(explainer.expected_value, shap_values[0])

    
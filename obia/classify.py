import numpy as np
import shap

from PIL.Image import fromarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from obia.handlers import _write_geotiff


class ClassifiedImage:
    classified_image = None
    confusion_matrix = None
    report = None
    params = None
    shap_values = None
    crs = None
    transform = None

    def __init__(self, classified_image, confusion_matrix, report, shap_values, transform, crs, params):
        self.classified_image = classified_image
        self.report = report
        self.confusion_matrix = confusion_matrix
        self.shap_values = shap_values
        self.params = params
        self.transform = transform
        self.crs = crs

    def write_geotiff(self, output_path):
        _write_geotiff(self.classified_image, output_path, self.crs, self.transform)


def classify(image, segmented_image, training_classes,
             method='rf', test_size=0.5, compute_reports=False,
             compute_shap=False, **kwargs):
    shap_values = None
    x = training_classes.drop(['feature_class', 'geometry', 'segment_id'], axis=1)
    y = training_classes['feature_class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    if method == 'rf':
        classifier = RandomForestClassifier(**kwargs)
    elif method == 'mlp':
        classifier = MLPClassifier(**kwargs)
    else:
        raise ValueError('An unsupported classification algorithm was requested')

    classifier.fit(x_train, y_train)
    if compute_shap:
        explainer = None
        if isinstance(classifier, RandomForestClassifier):
            explainer = shap.TreeExplainer(classifier)
        elif isinstance(classifier, MLPClassifier):
            explainer = shap.KernelExplainer(classifier.predict_proba, x_train)

        shap_values = explainer(x_train)

    y_pred = classifier.predict(x_test)

    report = None
    cm = None
    if compute_reports:
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

    x_pred = segmented_image.segments.drop(['feature_class', 'geometry', 'segment_id'], axis=1, errors='ignore')
    scaler = StandardScaler()

    scaler.fit(x_pred)
    x_pred = scaler.transform(x_pred)

    y_pred_all = classifier.predict(x_pred)

    params = classifier.get_params()
    segment_ids = segmented_image.segments['segment_id'].to_list()

    classified_img = np.zeros((image.img_data.shape[0], image.img_data.shape[1]))

    for i, segment_id in enumerate(segment_ids):
        idx = np.argwhere(segmented_image._segments == segment_id)
        for j in idx:
            classified_img[j[0], j[1]] = y_pred_all[i]

    return ClassifiedImage(classified_img, cm, report, shap_values, image.transform, image.crs, params)

# todo: add CNN classifier. Follow procedure of https://www.mdpi.com/2072-4292/13/14/2709#. simply plot each segment and assign a class then classify each plotted segment. seems super inneficient, but maybe more powerful? probably not though. RF or MLP should be just as good... but maybe not.
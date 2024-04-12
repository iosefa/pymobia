import numpy as np
from PIL.Image import fromarray
from skimage.util import img_as_float
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


from obia.handlers import _write_geotiff


class ClassifiedImage:
    crs = None
    transform = None
    original_image = None
    classified_image = None
    confusion_matrix = None
    params = None

    def __init__(self, original_image, classified_image, confusion_matrix, params):
        self.original_image = original_image
        self.classified_image = classified_image
        self.confusion_matrix = confusion_matrix
        self.params = params

    def write_geotiff(self, output_path):
        _write_geotiff(self.classified_image, output_path, self.original_image.crs, self.original_image.transform)


def classify(segmented_image, training_classes,
             method='random forests', compute_cm=False,
             validation_labels=None, **kwargs):
    cm = None
    x_train = training_classes.drop(['feature_class', 'geometry'], axis=1)
    y_train = training_classes['feature_class']

    if method == 'random forests':
        classifier = RandomForestClassifier(**kwargs)
    else:
        raise ValueError('An unsupported classification algorithm was requested')

    classifier.fit(x_train, y_train)

    if compute_cm:
        if validation_labels is None:
            raise ValueError("validation_labels must be provided when compute_cm is True.")

        segment_statistics_subset = segmented_image.statistics[
            segmented_image.statistics['segment_id'].isin(validation_labels['segment_id'])]
        if len(segment_statistics_subset.index) == 0:
            raise ValueError('validation_labels do not overlap with the segmented imaged.')
        x_test_subset = segment_statistics_subset.drop(['segment_id'], axis=1)
        y_pred_subset = classifier.predict(x_test_subset)

        cm = confusion_matrix(validation_labels['label'], y_pred_subset)

    y_pred_all = classifier.predict(
    segmented_image.statistics.drop(['feature_class', 'geometry'], axis=1, errors='ignore'))

    params = classifier.get_params()
    segment_ids = segmented_image.statistics['segment_id'].to_list()
    classified_img = img_as_float(np.array(segmented_image.original_image.img)).copy()

    for i, segment_id in enumerate(segment_ids):
        idx = np.argwhere(segmented_image.segments == segment_id)
        for j in idx:
            classified_img[j[0], j[1], 0] = y_pred_all[i]
    clf = classified_img[:, :, 0] * 255

    min_val = np.min(clf)
    max_val = np.max(clf)
    norm_clf = (clf - min_val) / (max_val - min_val)
    im = fromarray((norm_clf * 255).astype(np.uint8))
    return ClassifiedImage(segmented_image.original_image, im, cm, params)

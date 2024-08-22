import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from obia.handlers.geotif import _write_geotiff


class ClassifiedImage:
    classified = None
    confusion_matrix = None
    report = None
    params = None
    shap_values = None
    crs = None
    transform = None

    def __init__(self, classified, confusion_matrix, report, shap_values, transform, crs, params):
        self.classified = classified
        self.report = report
        self.confusion_matrix = confusion_matrix
        self.shap_values = shap_values
        self.params = params
        self.transform = transform
        self.crs = crs

    def write_geotiff(self, output_path):
        _write_geotiff(self.classified, output_path, self.crs, self.transform)


def classify(segments, training_classes, acceptable_classes_gdf=None,
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

        shap_values = explainer.shap_values(x_train)

    y_pred = classifier.predict(x_test)

    report = None
    cm = None
    if compute_reports:
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

    x_pred = segments.drop(['feature_class', 'geometry', 'segment_id'], axis=1, errors='ignore')
    scaler = StandardScaler()

    scaler.fit(x_pred)
    x_pred = scaler.transform(x_pred)

    # Initialize predicted classes and prediction margins
    y_pred_all = np.full(x_pred.shape[0], None)
    prediction_margin = np.full(x_pred.shape[0], None)

    for idx, segment in segments.iterrows():
        acceptable_classes = None

        if acceptable_classes_gdf is not None:
            # Check intersection with acceptable_classes_gdf
            intersections = acceptable_classes_gdf[acceptable_classes_gdf.intersects(segment.geometry)]
            if not intersections.empty:
                # If there are intersections, get the list of acceptable classes
                acceptable_classes = intersections.iloc[0]['acceptable_classes']

        if acceptable_classes is not None:
            # Predict the class and filter by acceptable classes
            proba = classifier.predict_proba([x_pred[idx]])
            proba_df = pd.DataFrame(proba, columns=classifier.classes_)
            proba_df_filtered = proba_df[proba_df.columns.intersection(acceptable_classes)]
            y_pred_all[idx] = proba_df_filtered.idxmax(axis=1).values[0]
            top2_probs = np.partition(proba_df_filtered.values[0], -2)[-2:]
            prediction_margin[idx] = top2_probs[1] - top2_probs[0]
        else:
            # Predict the class without filtering
            proba = classifier.predict_proba([x_pred[idx]])
            y_pred_all[idx] = classifier.predict([x_pred[idx]])[0]
            top2_probs = np.partition(proba[0], -2)[-2:]
            prediction_margin[idx] = top2_probs[1] - top2_probs[0]

    params = classifier.get_params()

    segments['predicted_class'] = y_pred_all
    segments['prediction_margin'] = prediction_margin

    for col in segments.columns:
        if col != segments.geometry.name:
            if np.issubdtype(segments[col].dtype, np.integer):
                segments[col] = segments[col].astype(pd.Int64Dtype())
            elif np.issubdtype(segments[col].dtype, np.floating):
                segments[col] = segments[col].astype(float)

    segments['predicted_class'] = segments['predicted_class'].astype(pd.Int64Dtype())
    segments['prediction_margin'] = segments['prediction_margin'].astype(float)

    return ClassifiedImage(segments, cm, report, shap_values, None, None, params)

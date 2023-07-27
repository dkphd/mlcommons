from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    matthews_corrcoef, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score, \
    ConfusionMatrixDisplay

from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import numpy as np
import matplotlib.pyplot as plt


class BaseClassificationMetrics:
    """
    Base metrics class for classification tasks.
    """

    def __init__(self, y_true, y_pred, y_prob=None):
        """
        Initialize with ground truth, predicted labels, and predicted probabilities.

        Parameters:
        - y_true (array-like): Ground truth labels.
        - y_pred (array-like): Predicted labels.
        - y_prob (array-like, optional): Predicted probabilities.
        """

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def accuracy(self):
        """
        Compute accuracy.

        Returns:
        float: Accuracy score.
        """

        """Compute accuracy"""
        return accuracy_score(self.y_true, self.y_pred)


class BinaryClassificationMetrics(BaseClassificationMetrics):

    def precision(self):
        """Compute precision for binary classification"""
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        """Compute recall for binary classification"""
        return recall_score(self.y_true, self.y_pred)

    def f1(self):
        """Compute F1 score for binary classification"""
        return f1_score(self.y_true, self.y_pred)

    def auc_roc(self):
        """Compute AUC-ROC for binary classification"""
        if self.y_prob is None:
            raise ValueError("y_prob is required for AUC-ROC computation.")
        return roc_auc_score(self.y_true, self.y_prob)

    def specificity(self):
        """
        Compute specificity (True Negative Rate) for binary classification.

        Returns:
        float: specificity.
        """
        tn, fp, _, _ = confusion_matrix(self.y_true, self.y_pred).ravel()

        if tn + fp == 0:
            return 0.0

        return tn / (tn + fp)

    def mcc(self):
        """Compute Matthews Correlation Coefficient for binary classification"""
        return matthews_corrcoef(self.y_true, self.y_pred)

    def precision_recall_auc(self):
        """
        Compute average Precision-Recall AUC for binary classification.
        """
        # For binary classification, use average_precision_score directly
        return average_precision_score(self.y_true, self.y_prob)


    def plot_roc_auc(self):
        """Return ROC-AUC Curve plot for binary classification"""
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        ax.plot(fpr, tpr, lw=2)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curve")
        ax.grid(alpha=0.2)
        return fig, ax

    def confusion_matrix(self):
        """Compute confusion matrix for binary classification"""
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_precision_recall_curve(self):
        """Return Precision-Recall Curve plot for binary classification"""
        fig, ax = plt.subplots()
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        ax.plot(recall, precision, lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_calibration_curve(self):
        """Return Calibration Curve plot for binary classification"""
        fig, ax = plt.subplots()
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_true, self.y_prob, n_bins=10)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-")

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_confusion_matrix(self):
        """
        Visualize the confusion matrix for binary classification using ConfusionMatrixDisplay without grid and colorbar.

        Returns:
        tuple: Matplotlib figure and axes objects.
        """
        cm = self.confusion_matrix()
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=['0', '1']).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.grid(False)
        plt.title('Confusion matrix')
        return fig, ax

    def calculate_numerical_metrics(self):
        """
        Compute and return all numerical metrics for binary classification.
        """
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'Specificity': self.specificity(),
            'F1 Score': self.f1(),
            'MCC': self.mcc(),
            'ROC AUC': self.auc_roc(),
            'Precision Recall AUC': self.precision_recall_auc()
        }
        return metrics

class MultiClassClassificationMetrics(BaseClassificationMetrics):

    def precision(self, average_method="macro"):
        """Compute precision for multiclass classification"""
        return precision_score(self.y_true, self.y_pred, average=average_method)

    def recall(self, average_method="macro"):
        """Compute recall for multiclass classification"""
        return recall_score(self.y_true, self.y_pred, average=average_method)

    def f1(self, average_method="macro"):
        """Compute F1 score for multiclass classification"""
        return f1_score(self.y_true, self.y_pred, average=average_method)

    def auc_roc(self):
        """Compute AUC-ROC for multiclass classification"""
        if self.y_prob is None:
            raise ValueError("y_prob is required for AUC-ROC computation.")
        return roc_auc_score(self.y_true, self.y_prob, average="macro", multi_class="ovr")

    def precision_recall_auc(self):
        """
        Compute average Precision-Recall AUC for multiclass classification.
        """
        # One-vs-all precision-recall curve for each class and then average the AUC values
        return average_precision_score(label_binarize(self.y_true, classes=np.unique(self.y_pred)), self.y_prob,
                                       average="macro")

    def plot_roc_auc(self):
        """Return ROC-AUC Curve plot for multiclass classification"""
        fig, ax = plt.subplots()
        n_classes = len(np.unique(self.y_true))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((self.y_true == i).astype(int), self.y_prob[:, i])
            ax.plot(fpr, tpr, lw=2, label=f'Class {i}')

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def specificity(self):
        """
        Compute specificity (True Negative Rate) for multiclass classification.

        Returns:
        float: Average specificity across classes.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        specificity_list = []

        # Calculate specificity for each class
        for i in range(cm.shape[0]):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[i])

            # Avoid division by zero
            if tn + fp == 0:
                specificity_list.append(0.0)
            else:
                specificity_list.append(tn / (tn + fp))

        return np.mean(specificity_list)

    def mcc(self):
        """
        Compute Matthews Correlation Coefficient for multiclass classification.

        Returns:
        float: MCC score.
        """
        # Flatten the arrays to compute MCC in multiclass scenario
        y_true_flat = self.y_true.argmax(axis=1)
        y_pred_flat = self.y_pred.argmax(axis=1)

        return matthews_corrcoef(y_true_flat, y_pred_flat)

    def confusion_matrix(self):
        """Compute confusion matrix for multiclass classification"""
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_precision_recall_curve(self):
        """Return Precision-Recall Curve plot for multiclass classification"""
        fig, ax = plt.subplots()
        n_classes = len(np.unique(self.y_true))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve((self.y_true == i).astype(int), self.y_prob[:, i])
            ax.plot(recall, precision, lw=2, label=f'Class {i}')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_calibration_curve(self):
        """Return Calibration Curve plot for multiclass classification"""
        fig, ax = plt.subplots()
        n_classes = len(np.unique(self.y_true))
        for i in range(n_classes):
            fraction_of_positives, mean_predicted_value = calibration_curve(self.y_true == i, self.y_prob[:, i],
                                                                            n_bins=10)
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'Class {i}')

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_confusion_matrix(self):
        """
        Visualize the confusion matrix for multiclass classification

        Returns:
        tuple: Matplotlib figure and axes objects.
        """
        cm = self.confusion_matrix()
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=['0', '1']).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.grid(False)
        plt.title('Confusion matrix')
        return fig, ax

    def calculate_numerical_metrics(self):
        """
        Compute and return all numerical metrics for multiclass classification.
        """
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'Specificity': self.specificity(),
            'F1 Score': self.f1(),
            'MCC': self.mcc(),
            'ROC AUC': self.auc_roc(),
            'Precision Recall AUC': self.precision_recall_auc()
        }
        return metrics


class MultiLabelClassificationMetrics(BaseClassificationMetrics):

    def precision(self, average_method="macro"):
        """Compute precision for multilabel classification"""
        return precision_score(self.y_true, self.y_pred, average=average_method)

    def recall(self, average_method="macro"):
        """Compute recall for multilabel classification"""
        return recall_score(self.y_true, self.y_pred, average=average_method)

    def f1(self, average_method="macro"):
        """Compute F1 score for multilabel classification"""
        return f1_score(self.y_true, self.y_pred, average=average_method)

    def auc_roc(self):
        """Compute AUC-ROC for multilabel classification"""
        if self.y_prob is None:
            raise ValueError("y_prob is required for AUC-ROC computation.")
        return roc_auc_score(self.y_true, self.y_prob, average="macro")

    def specificity(self):
        """
        Compute specificity (True Negative Rate) for multilabel classification.

        Returns:
        list: Specificity for each label.
        """
        specificity_list = []
        for i in range(self.y_true.shape[1]):
            tn, fp, _, _ = confusion_matrix(self.y_true[:, i], self.y_pred[:, i]).ravel()

            # Avoid division by zero
            if tn + fp == 0:
                specificity_list.append(0.0)
            else:
                specificity_list.append(tn / (tn + fp))

        return specificity_list

    def mcc(self):
        """Compute Matthews Correlation Coefficient for each label and average for multilabel classification"""
        mcc_per_label = [matthews_corrcoef(self.y_true[:, i], self.y_pred[:, i]) for i in range(self.y_true.shape[1])]
        return np.mean(mcc_per_label)

    def precision_recall_auc(self):
        """
        Compute average Precision-Recall AUC for multilabel classification.
        """
        # For multilabel classification, use average_precision_score directly with macro averaging
        return average_precision_score(self.y_true, self.y_prob, average="macro")

    def plot_roc_auc(self):
        """Return ROC-AUC Curve plot for multilabel classification"""
        fig, ax = plt.subplots()
        n_classes = self.y_true.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(self.y_true[:, i], self.y_prob[:, i])
            ax.plot(fpr, tpr, lw=2, label=f'Label {i}')

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_precision_recall_curve(self):
        """Return Precision-Recall Curve plot for multilabel classification"""
        fig, ax = plt.subplots()
        n_classes = self.y_true.shape[1]
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(self.y_true[:, i], self.y_prob[:, i])
            ax.plot(recall, precision, lw=2, label=f'Label {i}')

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def confusion_matrices(self):
        """Compute confusion matrix for each label in multilabel classification"""
        n_classes = self.y_true.shape[1]
        confusion_matrices = [confusion_matrix(self.y_true[:, i], self.y_pred[:, i]) for i in range(n_classes)]
        return confusion_matrices

    def plot_calibration_curve(self):
        """Return Calibration Curve plot for multilabel classification"""
        fig, ax = plt.subplots()
        n_classes = self.y_true.shape[1]
        for i in range(n_classes):
            fraction_of_positives, mean_predicted_value = calibration_curve(self.y_true[:, i], self.y_prob[:, i],
                                                                            n_bins=10)
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'Label {i}')

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_confusion_matrices(self):
        """
        Visualize the confusion matrices for multilabel classification, one for each label.

        Returns:
        tuple: Matplotlib figure and axes objects.
        """
        cms = self.confusion_matrices()  # List of confusion matrices for each label
        n_labels = len(cms)

        # Determine grid size based on the number of labels
        grid_size = int(np.ceil(np.sqrt(n_labels)))
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))

        # If only one label, axes is not a list, so we put it in a list for uniform handling
        if n_labels == 1:
            axes = [axes]

        # Plot each confusion matrix
        for i, ax in enumerate(axes.ravel()):
            if i < n_labels:
                ConfusionMatrixDisplay(cms[i], display_labels=['0', '1']).plot(cmap="Blues", ax=ax, colorbar=False)
                ax.set_title(f'Label {i}')
                ax.grid(False)
            else:
                # Turn off axes for extra subplots
                ax.axis('off')

        plt.suptitle('Confusion matrices for each label', y=1.02)
        plt.tight_layout()
        return fig, axes

    def calculate_numerical_metrics(self):
        """
        Compute and return all numerical metrics for multilabel classification.
        """
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'Specificity': self.specificity(),
            'F1 Score': self.f1(),
            'MCC': self.mcc(),
            'ROC AUC': self.auc_roc(),
            'Precision Recall AUC': self.precision_recall_auc()
        }
        return metrics

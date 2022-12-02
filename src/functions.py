import zipfile
import pandas as pd
import tensorflow as tf


class DrDataLoader(object):
    """Digit Recognizer competitiion dataset loader"""

    def __init__(self, dataset_fp):
        self.dataset_fp = dataset_fp

    def load_training_data(self):
        """Load training data."""
        archive = zipfile.ZipFile(self.dataset_fp, "r")
        features = pd.read_csv(archive.open("train.csv"))
        target = features.pop("label")
        return tf.data.Dataset.from_tensor_slices(
            (
                tf.reshape(tf.convert_to_tensor(features, tf.float32), [-1, 28, 28, 1]),
                target,
            )
        )

    def load_test_data(self):
        archive = zipfile.ZipFile(self.dataset_fp, "r")
        features = pd.read_csv(archive.open("test.csv"))
        return tf.data.Dataset.from_tensor_slices(
            (tf.reshape(tf.convert_to_tensor(features, tf.float32), [-1, 28, 28, 1]))
        )

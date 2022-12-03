from typing import Tuple
import zipfile
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


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


def prepare_pipeline(
    rescale_factor: float | None = None,
    rotation_factor: float | None = None,
    translation_factor: Tuple[float, float] | None = None,
    brightness_factor: float | None = None,
    zoom_factor: float | None = None,
):
    """Create a pipeline as a keras sequential model.

    Args:
        rescale_factor (float | None, optional): Rescale factor for images. Defaults to None.
        rotation_factor (float | None, optional): Random rotation factor of 2pi. Defaults to None.
        translation_factor (Tuple[float, float] | None, optional): Height, Width random translation factor. Defaults to None.
        brightness_factor (float | None, optional): Random brightness factor. Defaults to None.
        zoom_factor (float | None, optional): Random zoom factor. Defaults to None.
    
    Returns:
        List: List of data augmentation layers.

    """
    layers = []
    value_range = (0, 255.0)
    if rescale_factor is not None:
        layers.append(tf.keras.layers.Rescaling(scale=rescale_factor))
        value_range = tuple(i * rescale_factor for i in value_range)  # type: ignore

    if brightness_factor is not None:
        layers.append(
            tf.keras.layers.RandomBrightness(
                factor=brightness_factor, value_range=value_range
            ),
        )

    if rotation_factor is not None:
        layers.append(tf.keras.layers.RandomRotation(factor=rotation_factor))

    if translation_factor is not None:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=translation_factor[0], width_factor=translation_factor[1]
            )
        )

    if zoom_factor is not None:
        layers.append(tf.keras.layers.RandomZoom(zoom_factor))

    return layers


def prepare_ds(
    ds: tf.data.Dataset,
    rescale_factor: float | None = None,
    batch_size: int | None = None,
    rotation_factor: float | None = None,
    translation_factor: Tuple[float, float] | None = None,
    brightness_factor: float | None = None,
    zoom_factor: float | None = None,
):
    """Prepare data augmention pipeline

    Args:
        ds (tf.data.Dataset): Dataset to prepare.
        rescale_factor (float | None, optional): Rescale factor for images. Defaults to None.
        batch_size (int | None, optional): Batch size for pipeline. Defaults to None.
        rotation_factor (float | None, optional): Random rotation factor of 2pi. Defaults to None.
        translation_factor (Tuple[float, float] | None, optional): Height, Width random translation factor. Defaults to None.
        brightness_factor (float | None, optional): Random brightness factor. Defaults to None.
        zoom_factor (float | None, optional): Random zoom factor. Defaults to None.

    Returns:
        tf.data.Dataset: Dataset with data auugmentation.
    """

    pipeline = tf.keras.Sequential(
        prepare_pipeline(
            rescale_factor=rescale_factor,
            rotation_factor=rotation_factor,
            translation_factor=translation_factor,
            brightness_factor=brightness_factor,
            zoom_factor=zoom_factor,
        )
    )

    ds = ds.map(lambda x, y: (pipeline(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds

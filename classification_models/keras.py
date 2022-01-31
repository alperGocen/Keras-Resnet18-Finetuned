import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()
import tensorflow.keras as keras
from .models_factory import ModelsFactory


class KerasModelsFactory(ModelsFactory):

    @staticmethod
    def get_kwargs():
        return {
            'backend': keras.backend,
            'layers': keras.layers,
            'models': keras.models,
            'utils': keras.utils,
        }


Classifiers = KerasModelsFactory()

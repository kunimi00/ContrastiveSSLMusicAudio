from abc import ABC, abstractmethod

class AbstractModel(ABC):
    """
    Abstract base class for ML/DL model
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization
        """
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        preprocess
        """
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """
        postprocess
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        After applying preprocess, fit the model
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        After applying preprocess, predict with the final estimator
        """
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """
        Load the model
        """
        pass
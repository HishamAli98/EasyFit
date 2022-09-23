from .models import DefaultModel, CLASSIFIERS


class DefaultClassifier(DefaultModel):

    def __init__(self):
        super().__init__(CLASSIFIERS)

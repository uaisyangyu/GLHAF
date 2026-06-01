from .singleTask import *

__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'G2L': G2L,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
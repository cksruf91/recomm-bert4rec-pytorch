import os


class Config:
    ROOT = os.path.join('.')
    DATA = os.path.join(ROOT, 'datasets')


CONFIG = Config()

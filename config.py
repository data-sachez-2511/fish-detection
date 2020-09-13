import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.num_epochs = 20
Cfg.lr = 0.001
Cfg.models_dir = os.path.join(_BASE_DIR, 'weights')
Cfg.confidence_score = 0.8
Cfg.num_boxes = 10

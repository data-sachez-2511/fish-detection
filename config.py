import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.num_epochs = 20
Cfg.lr = 0.001
Cfg.models_dir = os.path.join(_BASE_DIR, 'weights')
Cfg.confidence_score = 0.8
Cfg.num_boxes = 10
Cfg.train_dataset = 'D:\\Data\\fish\\datasets\\fish_in_the_wild\\train'
Cfg.test_dataset = 'D:\\Data\\fish\\datasets\\fish_in_the_wild\\test'

Cfg.train_labels = 'D:\\Data\\fish\\datasets\\fish_in_the_wild\\train.txt'
Cfg.test_labels = 'D:\\Data\\fish\\datasets\\fish_in_the_wild\\test.txt'

Cfg.variance = [0.1, 0.2]
Cfg.feature_maps = [38, 19, 10, 5, 3, 1]
Cfg.min_dim = 300
Cfg.steps = [8, 16, 32, 64, 100, 300]
Cfg.min_sizes = [30, 60, 111, 162, 213, 264]
Cfg.max_sizes = [60, 111, 162, 213, 264, 315]
Cfg.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
Cfg.clip = True
Cfg.name = 'VOC'

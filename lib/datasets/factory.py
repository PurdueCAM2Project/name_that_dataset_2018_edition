# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.inria import inria
from datasets.kitti import kitti
from datasets.caltech import caltech
from datasets.sun import sun
from datasets.cam2 import cam2
from datasets.pascal_voc import pascal_voc
from datasets.imagenet import imagenet
from datasets.coco import coco
import numpy as np

# Set up imagenet_<year>_<split>
for year in ['2014']:
    for split in ['train', 'val', 'val1', 'val2', 'test','short_train','very_short_train','val1_short']:
        name = 'imagenet_{}'.format(split)
        __sets[name] = (lambda split=split: imagenet(split))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test','val_small']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up cam2_2017_<split>
for year in ['2017']:
    for split in ['train','val','trainval','test','all']:
        name = 'cam2_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: cam2(split, year))

# Set up sun_2012_<split>
for year in ['2012']:
    for split in ['taste','all','test','train']:
        name = 'sun_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: sun(split, year))

# Set up caltech_2009_<split>
for year in ['2009']:
    for split in ['val','train','test','all','taste','medium']:
        name = 'caltech_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: caltech(split, year))

# Set up kitti_2013_<split> # TODO
for year in ['2013']:
    for split in ['val','train','test','all']:
        name = 'kitti_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: kitti(split, year))

# Set up inria_2005_<split> # TODO
for year in ['2005']:
    for split in ['val','train','test','all']:
        name = 'inria_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: inria(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

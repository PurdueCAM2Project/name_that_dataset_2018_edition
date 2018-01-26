# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

class index(object):
    """Image database."""

    def __init__(self):
        pass

    def _load_image_set_index(self):
        pass


class i_sun(index):

    def __init__(self, index_info, dataset):
        index.__init__(self)
        self._index = index_info
        self._image_path = dataset._image_path

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

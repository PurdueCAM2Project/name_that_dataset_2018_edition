# --------------------------------------------------------
# Fast R-CNN (addition)
# Written by Kent Gauen
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

# we want:
# -> a list of indexes for shift through

class unite(imdb):

    def __init__(self,imdb_list):
        self.imdb_list = imdb_list
        self.gt_roidb = []
        self.gt_classes = []
        self.virtual_index = []
        self._create_gt_roidb()
        self._num_gt = len(self.gt_roidb)
        self._randomize_gt_roidb()
        self.num_classes = len(imdb_list)
        self.classes = keys(self.imdb_dict)
        self.name = "united"

    def _create_roidb_to_class(self):
        self.imdb_dict = {}
        for imdb in imdb_list: 
            self.imdb_dict += {imdb.dataset_name:imdb}
        
    def _create_gt_roidb(self):
        for imdb in self.imdb_list:
            roidb = imdb.roidb() 
            self.gt_roidb = self.gt_roidb + roidb
            self.gt_classes = self.gt_classes +\
                              [imdb.dataset_name for _ in range(len(roidb))]
            self.virtual_index = self.virtual_index + imdb.image_index

    def _randomize_gt_roidb(self):
        indicies = np.random.permutation(self._num_gt)
        self.gt_roidb = [self.gt_roidb[ix] for ix in indicies]
        self.gt_classes = [self.gt_classes[ix] for ix in indicies]
        self.virtual_index = [self.virtual_index[ix] for ix in indicies]

    def gt_roidb(self):
        return self.gt_roidb

    def image_path_at(self, i):
        # note that "i" is an "index" class
        imdb = self.imdb_dict[i.dataset]
        return imdb.image_path_from_index(imdb.image_index[self.virtual_index[i.index]])
        


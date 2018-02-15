# --------------------------------------------------------
# Fast R-CNN (addition)
# Written by Kent Gauen
# --------------------------------------------------------

import os,sys,cPickle,PIL,uuid
from datasets.imdb import imdb
import os.path as osp
from fast_rcnn.train import get_training_roidb
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

class unite(imdb):

    def __init__(self,imdb_list,num_per_set,shuffle=False,shuffled_set=None):
        self.imdb_list = imdb_list
        self.num_per_set = num_per_set
        self._roidb = None
        self._roidb_handler = self.gt_roidb
        self._gt_classes = [] # used for virtual indexing
        self._virtual_index = [] # maps the indicies from the specific dataset to the united roidb
        self._create_roidb_to_class()
        self._num_gt = 0

        # handling shuffling the images
        self._shuffled = None
        if shuffle is True:
            if shuffled_set is None:
                self._shuffle_sets_and_save()
            else:
                self._load_shuffled_sets(shuffled_set)

        self._classes = []
        for cls in self.imdb_dict.keys():
            # handle the voc specially, there are two of them
            if "voc" in cls and "voc" not in self._classes:
                self._classes += ["voc"]
            elif "voc" not in cls:
                self._classes += [cls]

        self._randomize_gt_roidb()

        self._name = "united"

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_dict = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            self._virtual_index = gt_dict['vi']
            self._gt_classes = gt_dict['gt_classes']
            return gt_dict['gt_roidb']
        
        gt_dict = self._create_gt_roidb_fixed()
        self._virtual_index = gt_dict['vi']
        self._gt_classes = gt_dict['gt_classes']

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_dict, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_dict['gt_roidb']

    def _shuffle_sets_and_save(self):
        """
        Shuffle the indicies for each dataset and save them to file
        """
        self._shuffled = {}
        for imdb in self.imdb_list: 
            self._shuffled[imdb.dataset_name] = np.random.permutation(len(imdb.roidb))[:self.num_per_set]
        filename = osp.join(cfg.SHUFFLE_DIR,"{}".format(str(uuid.uuid4())))
        print("writing shuffled indicies to {}".format(filename))
        fid = open(filename,"w")
        cPickle.dump(self._shuffled,fid)
        
    def _load_shuffled_sets(self,filename):
        """
        Load in the shuffled indicies for each dataset
        """
        self._shuffled = cPickle.load(filename)

    def _create_roidb_to_class(self):
        """
        Create roidb for classes
        """
        self.imdb_dict = {}
        for imdb in self.imdb_list: 
            imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
            self.imdb_dict[imdb.dataset_name] = imdb
        
    def _create_gt_roidb_fixed(self):
        """
        -> Init the roidbs for the imdbs and take a fixed number
        -> Make a list of the roidb images and classes
        """

        # take a random ordering if loaded.
        # TODO: make the dictionary for idx when shuffled == None
        gt_roidb = []
        gt_classes = []
        virtual_index = []

        if self._shuffled is not None:
            idx = self._shuffled
        else:
            idx = np.arange(self.num_per_set)

        for imdb in self.imdb_list:
            roidb = imdb.roidb
            gt_classes += [imdb.dataset_name for _ in range(self.num_per_set)]
            if self._shuffled is not None:
                assert len(self._shuffled[imdb.dataset_name]) == self.num_per_set, "The shuffled indicies need to be the same size as the number of loaded datums per image set"
                gt_roidb += [roidb[i] for i in self._shuffled[imdb.dataset_name]]
                virtual_index += [imdb.image_index[i] for i in self._shuffled[imdb.dataset_name]]
            else:
                gt_roidb += roidb[:self.num_per_set]
                virtual_index += imdb.image_index[:self.num_per_set]
        return {"gt_roidb":gt_roidb,"gt_classes":gt_classes,"vi":virtual_index}

    def _randomize_gt_roidb(self):
        if self._num_gt == 0:
            print("Warning: can't randomize the roidb's yet... we don't have them...")
            return
        indicies = np.random.permutation(self._num_gt)
        self._roidb = [self._roidb[ix] for ix in indicies]
        self._gt_classes = [self._gt_classes[ix] for ix in indicies]
        self._virtual_index = [self._virtual_index[ix] for ix in indicies]

    # def gt_roidb(self):
    #     return self.gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # note that "i" is an "index" class
        imdb = self.imdb_dict[self._gt_classes[i]]
        return imdb.image_path_from_index(imdb.image_index[self._virtual_index[i.index]])
        


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os,sys
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from imagenet_eval import voc_eval
from fast_rcnn.config import cfg

class imagenet(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'imagenet_' + image_set)
        self._year = "2014"
        self._image_set = image_set
        if "val" in image_set:
            self._image_set_dir = "val"
            if "val1" in image_set:
                self._anno_set_dir = "val1"
            if "val2" in image_set:
                self._anno_set_dir = "val2"
        elif "train" in image_set:
            self._image_set_dir = "train"
            self._anno_set_dir = "train"
        else:
            self._image_set_dir = image_set
            self._anno_set_dir = image_set

        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path)
	synsets = sio.loadmat(os.path.join(self._devkit_path, 'meta_det.mat'))
        self._classes = ('__background__',)
	self._wnid = (0,)

        # Noramlly set to loop below
	for i in xrange(200):
		self._classes = self._classes + (synsets['synsets'][0][i][2][0],)
		self._wnid = self._wnid + (synsets['synsets'][0][i][1][0],)

        # set to new loop so we can use the VOC weights
	# for i in xrange(20):
        #     if i == 14:
	#         self._classes = self._classes + (synsets['synsets'][0][123][2][0],)
	#         self._wnid = self._wnid + (synsets['synsets'][0][123][1][0],)
        #     else:
	#         self._classes = self._classes + (synsets['synsets'][0][i][2][0],)
	#         self._wnid = self._wnid + (synsets['synsets'][0][i][1][0],)
	self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._ind_to_wnid = dict(zip(xrange(self.num_classes),self._wnid))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # ImageNet specific config options (Kent :D)
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'matlab_eval' : False,
                       'use_diff'    : False,
                       'rpn_file'    : None,
                       'min_size' : 2}

        assert os.path.exists(self._devkit_path), \
                'ImageNet path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data/DET/',self._image_set_dir,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets/', 'DET',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split()[0] for x in f.readlines()]
            #image_index = [x.strip() for x in f.readlines()]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where ImageNet is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ImageNet')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations/DET', self._image_set_dir,index + '.xml')
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data
        with open(filename) as f:
            data = minidom.parseString(f.read())
        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._wnid_to_ind[str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'det_results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} ImageNet results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            'DET',
            self._image_set_dir,
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            'DET',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache',self._anno_set_dir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            # pass in the class id corresponding to the synset
            n_cls = self._wnid[i]
            rec, prec, ap,ovthresh = voc_eval(
                filename, annopath, imagesetfile, n_cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            #print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        #print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print(aps)
        aps = np.array(aps)
        results_fd = open("./results_imagenet.txt","w")
        for kdx in range(len(ovthresh)):
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            print('Mean AP = {:.4f} @ {:.2f}'.format(np.mean(aps[:,kdx]),ovthresh[kdx]))
        print('~~~~~~~~')
        print('Results:')
        sys.stdout.write('{0:>15} (#):'.format("class AP"))
        results_fd.write('{0:>15} (#):'.format("class AP"))
        for thsh in ovthresh:
            sys.stdout.write("\t{:>5}{:.3f}".format("@",thsh))
            results_fd.write("\t{:>5}{:.3f}".format("@",thsh))
        sys.stdout.write("\n")
        results_fd.write("\n")
        count_ = 1
        for ap in aps:
            sys.stdout.write('{:>15} ({}):'.format(self._classes[count_],count_))
            results_fd.write('{:>15} ({}):'.format(self._classes[count_],count_))
            for kdx in range(len(ovthresh)):
                sys.stdout.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
                results_fd.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
            sys.stdout.write('\n')
            results_fd.write('\n')
            count_ +=1
        sys.stdout.write('{:>15}:'.format("mAP"))
        results_fd.write('{:>15}:'.format("mAP"))
        for kdx in range(len(ovthresh)):
            sys.stdout.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            results_fd.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            #print('mAP @ {:.2f}: {:.5f} '.format(ovthresh[kdx],np.mean(aps[:,kdx])))
        sys.stdout.write('\n')
        results_fd.write('\n')
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.imagenet import imagenet
    d = imagenet('train', '2014')
    res = d.roidb
    from IPython import embed; embed()

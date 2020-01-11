# -*- coding: utf-8 -*-
import json
import os
import sys
import time
import zipfile
from collections import OrderedDict
from functools import cmp_to_key
from pathlib import Path

from PIL import Image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage import io
from torch.autograd import Variable

from .craft import craft_utils, imgproc
from .craft.craft import CRAFT

DATASET = (Path(__file__).parent / '..' / 'data').resolve()


def _copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    # end if
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    # end for
    return new_state_dict
# end def


class Detector:
    cuda = False
    canvas_size = 1280
    magnify_ratio = 1.5
    text_threshold = 0.7
    link_threshold = 0.4
    low_text_score = 0.4
    enable_ploygon = False
    enable_refiner = False
    trained_model = str(DATASET / 'craft_mlt_25k.pth')
    refiner_model = str(DATASET / 'craft_refiner_CTW1500.pth')

    def load(self):
        self.net = CRAFT()     # initialize

        if torch.cuda.is_available():
            self.cuda = True
            self.net.load_state_dict(_copyStateDict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(_copyStateDict(
                torch.load(self.trained_model, map_location='cpu')))
        # end if

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        # end if

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if self.enable_refiner:
            from .craft.refinenet import RefineNet
            self.refine_net = RefineNet()
            if self.cuda:
                self.refine_net.load_state_dict(_copyStateDict(torch.load(self.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(_copyStateDict(
                    torch.load(self.refiner_model, map_location='cpu')))
            # end if

            self.refine_net.eval()
            self.enable_ploygon = True
        # end if
    # end def

    def process(self, image):
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.magnify_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()
        # end if

        # forward pass
        y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
        # end if

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text,
                                               score_link,
                                               self.text_threshold,
                                               self.link_threshold,
                                               self.low_text_score,
                                               self.enable_ploygon)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
            # end if
        # end for

        # transform boxes to top-left and bottom-right points
        rects = list()
        for box in boxes:
            poly = np.array(box).astype(np.int32)
            y0, x0 = np.min(poly, axis=0)
            y1, x1 = np.max(poly, axis=0)
            rects.append([x0, y0, x1, y1])
        # end for

        # comparator to sort rectangles in LTR text mode
        def compare_rect(first_rect, second_rect):
            fx, fy, fxi, fyi = first_rect
            sx, sy, sxi, syi = second_rect
            if fxi <= sx:
                return -1  # completely on above
            elif sxi <= fx:
                return 1    # completely on below
            elif fyi <= fy:
                return -1  # completely on left
            elif sxi <= sx:
                return 1  # completely on right
            elif fy != sy:
                return -1 if fy < sy else 1  # starts on more left
            elif fx != sx:
                return -1 if fx < sx else 1  # top most when starts equally
            elif fyi != syi:
                return -1 if fyi < syi else 1  # have least width
            elif fxi != sxi:
                return -1 if fxi < sxi else 1  # have laast height
            else:
                return 0  # same
            # end if
        # end def

        # extract ROIs
        roi = list()
        for rect in sorted(rects, key=cmp_to_key(compare_rect)):
            x0, y0, x1, y1 = rect
            sub = image[x0:x1, y0:y1, :]
            roi.append(sub)
        # end for

        return roi, boxes, polys, image
    # end def
# end class

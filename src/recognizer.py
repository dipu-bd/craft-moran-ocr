from collections import OrderedDict
from pathlib import Path

from PIL import Image

import cv2
import torch
from torch.autograd import Variable

from .moran.models.moran import MORAN
from .moran.tools import dataset, utils

DATASET = (Path(__file__).parent / '..' / 'data').resolve()


class Recognizer:
    model_path = str(DATASET / 'moran_v2_demo.pth')
    alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
    max_iter = 20
    cuda = False
    moran = None
    state_dict = None
    converter = None
    transformer = None

    def load(self):
        if torch.cuda.is_available():
            self.cuda = True
            self.moran = MORAN(1, len(self.alphabet.split(':')), 256,
                               32, 100, BidirDecoder=True, CUDA=True)
            self.moran = self.moran.cuda()
        else:
            self.moran = MORAN(1, len(self.alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                               inputDataType='torch.FloatTensor', CUDA=False)
        # end if

        print('Loading pretrained model from %s' % self.model_path)
        if self.cuda:
            self.state_dict = torch.load(self.model_path)
        else:
            self.state_dict = torch.load(self.model_path, map_location='cpu')
        # end if

        MORAN_state_dict_rename = OrderedDict()
        for k, v in self.state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        # end for
        self.moran.load_state_dict(MORAN_state_dict_rename)

        for p in self.moran.parameters():
            p.requires_grad = False
        # end for
        self.moran.eval()

        self.converter = utils.strLabelConverterForAttention(self.alphabet, ':')
        self.transformer = dataset.resizeNormalize((100, 32))
    # end def

    def process(self, cv_image):
        image = Image.fromarray(cv_image).convert('L')
        image = self.transformer(image)
        if self.cuda:
            image = image.cuda()
        # end if

        image = image.view(1, *image.size())
        image = Variable(image)
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        text = Variable(text)
        length = Variable(length)

        t, l = self.converter.encode('0'*self.max_iter)
        utils.loadData(text, t)
        utils.loadData(length, l)
        output = self.moran(image, length, text, text, test=True, debug=True)

        preds, preds_reverse = output[0]
        demo = output[1]

        _, preds = preds.max(1)
        _, preds_reverse = preds_reverse.max(1)

        sim_preds = self.converter.decode(preds.data, length.data)
        sim_preds = sim_preds.strip().split('$')[0]
        sim_preds_reverse = self.converter.decode(preds_reverse.data, length.data)
        sim_preds_reverse = sim_preds_reverse.strip().split('$')[0]

        return sim_preds, sim_preds_reverse, demo
    # end def
# end class

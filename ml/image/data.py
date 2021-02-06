import pandas as pd
import numpy as np
from fastai.vision import *

""" A Databunch is used in FastAI to wrap datasets for training and validation. A Databunch should provide dataloaders for both via train_dl and valid_dl.

Here we use the FastAI vision databunch, which takes a path in which the images are stored in folders that indicate their class.
""" 

def image_databunch(config, path, size, valid_pct=0.2, tfms=None):
    if tfms == None:
        tfms = get_transforms()
    return ImageList.from_folder(path)\
            .split_by_rand_pct(valid_pct=valid_pct)\
            .label_from_folder()\
            .transform(tfms, size=size)\
            .databunch(bs = config.batch_size, num_workers = config.num_workers, device = config.device)
    #return ImageDataBunch.from_folder(path, ds_tfms=tfms, valid='valid', test='test', size=size, bs = config['batch_size'], device = config['device'])

def text_image_databunch(config, path, size, valid_pct=0.2):
    """ Specific for text images, we have to get rid of the flip transformation, because it does not apply to text 
    """
    tfms = get_transforms()
    tfms = tuple([t for i, t in enumerate(tr) if i != 1] for tr in tfms)

    return image_databunch(config, path, size, tfms=tfms)


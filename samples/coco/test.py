import os

import tensorflow as tf
from tensorflow import keras
import coco
from pycocotools.coco import COCO

class CarsConfig(Config):
    NAME = "cars"
    NUM_CLASSES = 1 + 1 # background + cars

config = CarsConfig()  # Don't forget to use this config while creating your model
config.display()

ct = COCO("/YourPathToCocoDataset/annotations/instances_train2014.json")
ct.getCatIds(['sheep']) 
# Sheep class' id is 20. You should run for person and use that id

COCO_DIR = "/YourPathToCocoDataset/"
# This path has train2014, annotations and val2014 files in it

# Training dataset
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", class_ids=[3])
dataset_train.prepare()

# Validation dataset
dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, "val", class_ids=[3])
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# This COCO_MODEL_PATH is the path to the mask_rcnn_coco.h5 file in this repo

model.train(dataset_train, dataset_val,
    learning_rate=config.LEARNING_RATE, 
    epochs=100, 
    layers='heads')#You can also use 'all' to train all network.
    
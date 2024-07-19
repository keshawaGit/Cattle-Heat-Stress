# https://youtu.be/QntADriNHuk
"""
Mask R-CNN - Multiclass - Coco style annotations in JSON format

For annotations, use one of the following programs: 
    https://www.makesense.ai/
    https://labelstud.io/
    https://github.com/Doodleverse/dash_doodler
    http://labelme.csail.mit.edu/Release3.0/
    https://github.com/openvinotoolkit/cvat
    https://www.robots.ox.ac.uk/~vgg/software/via/
    
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw

from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle
#########################

class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids




##############################

dataset_train = CocoLikeDataset()
dataset_train.load_data('cattle_dataset/train/_annotations.coco.json', 'cattle_dataset/train')
dataset_train.prepare()

#In this example, I do not have annotations for my validation data, so I am loading train data
dataset_val = CocoLikeDataset()
dataset_val.load_data('cattle_dataset/valid/_annotations.coco.json', 'cattle_dataset/valid')
dataset_val.prepare()


dataset = dataset_train
image_ids = dataset.image_ids
#image_ids = np.random.choice(dataset.image_ids, 3)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    display_top_masks(image, mask, class_ids, dataset.class_names, limit=2)  #limit to total number of classes



# define image id
image_id = 0
# load the image
image = dataset_train.load_image(image_id)
# load the masks and the class ids
mask, class_ids = dataset_train.load_mask(image_id)

# display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# define a configuration for the model
class CattleConfig(Config):
	# define the name of the configuration
	NAME = "cattle_cfg_coco"
	# number of classes (background + cattle sitting + cattle laying)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 100
    #DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence
# prepare config
config = CattleConfig()
config.display() 




ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "coco_weights/mask_rcnn_coco.h5")

########################
#Weights are saved to root D: directory. need to investigate how they can be
#saved to the directory defined... "logs_models"

###############

# define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads')


###################################################
# # define the prediction configuration
# class PredictionConfig(Config):
# 	# define the name of the configuration
# 	NAME = "cattle_cfg_coco"
# 	# number of classes (background + cattle sitting + cattle laying)
# 	NUM_CLASSES = 1 + 1
# 	# Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
# 	GPU_COUNT = 1
# 	IMAGES_PER_GPU = 1
# 	USE_MINI_MASK = False
   
# # calculate the mAP for a model on a given dataset
# def evaluate_model(dataset, model, cfg):
# 	APs = list()
# 	for image_id in dataset.image_ids:
# 		# load image, bounding boxes and masks for the image id
# 		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id) #, use_mini_mask=False
# 		# convert pixel values (e.g. center)
# 		scaled_image = mold_image(image, cfg)
# 		# convert image into one sample
# 		sample = expand_dims(scaled_image, 0)
# 		# make prediction
# 		yhat = model.detect(sample, verbose=0)
# 		# extract results for first sample
# 		r = yhat[0]
# 		# calculate statistics, including AP
# 		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
# 		# store
# 		APs.append(AP)
# 	# calculate the mean AP across all images
# 	mAP = mean(APs)
# 	return mAP
 

# # create config
# cfg = PredictionConfig()
# # define the model
# model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# # load model weights
# model.load_weights('logs/cattle_cfg_coco20230416T0236/mask_rcnn_cattle_cfg_coco_0025.h5', by_name=True)
# # evaluate model on training dataset
# train_mAP = evaluate_model(dataset_train, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# # evaluate model on test dataset
# # test_mAP = evaluate_model(dataset_train, model, cfg)
# # print("Test mAP: %.3f" % test_mAP)
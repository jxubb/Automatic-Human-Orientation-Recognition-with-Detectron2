# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
from PIL import Image
from operator import itemgetter


# create config
cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.DEVICE = "cpu"


from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from starlette.responses import FileResponse
from typing import List
app = FastAPI()


# Rotate image
def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


@app.post("/predict_front_RotNet_detectron2/")
async def predict_batch_front_RotNet_detectron2(input_images: List[UploadFile] = File(...)):
    """Detect the orientation of the front of a batch of sports cards"""
    predicted_angles = []
    # create predictor
    predictor = DefaultPredictor(cfg)
    for input_image in input_images:
        if input_image.content_type.startswith('image/') is False:
            raise HTTPException(status_code=400, detail=f'File \'{input_image.filename}\' is not an image.')
        input_image = Image.open(input_image.file)
        input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
        
        sum_conf = []
        try:
            # make prediction
            output = predictor(input_image)["instances"]
            key_points = output.get_fields()["pred_keypoints"].tolist()
            sum_conf.append( sum([key_points[0][i][2]  for i in range(len(key_points[0][:]))]) )
        
            output = predictor(rotate(input_image, -90))["instances"]
            key_points = output.get_fields()["pred_keypoints"].tolist()
            sum_conf.append( sum([key_points[0][i][2]  for i in range(len(key_points[0][:]))]) )
        
            output = predictor(rotate(input_image, -180))["instances"]
            key_points = output.get_fields()["pred_keypoints"].tolist()
            sum_conf.append( sum([key_points[0][i][2]  for i in range(len(key_points[0][:]))]) )
        
            output = predictor(rotate(input_image, -270))["instances"]
            key_points = output.get_fields()["pred_keypoints"].tolist()
            sum_conf.append( sum([key_points[0][i][2]  for i in range(len(key_points[0][:]))]) )
        
            index, element = max(enumerate(sum_conf), key = itemgetter(1))
            predicted_angle = np.int(index*90)
        except:
            predicted_angle = "Key point missing" 
        predicted_angles.append(predicted_angle)
     
    return dict(zip([input_image.filename for input_image in input_images],predicted_angles))
        
         

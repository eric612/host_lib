"""
This is the 720 public Yolov3 example with postprocessing done on host side.
"""
import pathlib
import cv2
from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_
from common.pre_post_process.kneron_pre.kneron_preprocess import *

# Example model/image paths
ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent.parent

ANCHOR_FOLDER = ROOT_FOLDER / "python/common/pre_post_process/yolo/models"
ANCHOR_PATH = str(ANCHOR_FOLDER / "anchors.txt")

MODEL_FOLDER = ROOT_FOLDER / "input_models/KL720/yolov3_bdd100k"
MODEL_PATH = str(MODEL_FOLDER / "models_720.nef")

CLASS_FOLDER = ROOT_FOLDER / "python/common/class_lists"
CLASS_PATH = str(CLASS_FOLDER / "bdd100k_list")

# RGB565 input image configurations
IMAGE_CHANNEL = 3
IMAGE_WIDTH = 352
IMAGE_HEIGHT = 352
IMAGE_BPP = 2
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BPP
# RGB565 input binary, radix = 7 for YOLO model, RAW_OUTPUT for host postproces
IMAGE_FORMAT = (constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_SUB128 | 
                constants.IMAGE_FORMAT_RAW_OUTPUT)

# Postprocess parameters
MODEL_HEIGHT = 352
MODEL_WIDTH = 352
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
# App ID of APP_CENTER_APP will always works for a single model
APP_ID = constants.AppID.APP_CENTER_APP.value

# Model ID is the same one generated with batch compile (32768 in this case)
MODEL_ID = constants.ModelType.CUSTOMER.value

    
def user_test_single_yolo(device_index):
    """Test single ISI."""
    # Load NEF model into board.
    isi_data = kdp_wrapper.init_isi_data(
        IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_FORMAT, MODEL_ID)
    ret = kdp_wrapper.isi_load_nef(device_index, MODEL_PATH, APP_ID, isi_data=isi_data)
    
    if ret:
        return ret
    # Setup video capture device.
    cap = kdp_wrapper.setup_capture("../video/d7ae13cc-05b75de8.mov", IMAGE_WIDTH, IMAGE_HEIGHT)
    if cap is None:
        return -1
    frames = []
    keep_aspect_ratio = False
    img_id = 1


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    with open(CLASS_PATH) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (DISPLAY_WIDTH,  DISPLAY_HEIGHT))
    
    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret==False:
            break
        #frame = cv2.transpose(frame)
        #frame = cv2.flip(frame,0)
        #frames.append(frame)
        frame2 = cv2.resize(frame, (IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

        # Inference the image.
        img_buf = kdp_wrapper.convert_numpy_to_char_p(frame2,color = cv2.COLOR_BGR2BGR565, size=IMAGE_SIZE)
        img_left = kdp_wrapper.isi_inference(device_index, img_buf, IMAGE_SIZE, img_id)

        if img_left == -1:
            return -1       
        result_data, _result_size = kdp_wrapper.isi_retrieve_res(device_index, img_id)
        if result_data is None:
            return -1
        # Output will be in (1, h, w, c) format
        np_results = kdp_wrapper.convert_data_to_numpy(result_data, add_batch=True, channel_last=True)

        dets = yolo_postprocess_(np_results, ANCHOR_PATH, CLASS_PATH, DISPLAY_HEIGHT, DISPLAY_WIDTH,
                                 (MODEL_WIDTH, MODEL_HEIGHT), SCORE_THRESHOLD, NMS_THRESHOLD, False)
        #print(dets[0])            
        img_id += 1    
        
        if ret == True:
            frame = cv2.resize(frame, (DISPLAY_WIDTH,DISPLAY_HEIGHT),interpolation=cv2.INTER_AREA)        
            for idx,bbox in enumerate(dets):
                #print(bbox)
                x1,y1,x2,y2,cls_idx = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                cv2.putText(frame,class_names[cls_idx],(x1,y1-5),0,0.3,(0,0,255))       
            cv2.imshow('Frame',frame)
            out.write(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break       
    cap.release()
    out.release()
    cv2.destroyAllWindows()    

    return 0

def user_test(device_index, _user_id):
    """ISI test."""
    ret = user_test_single_yolo(device_index)
    kdp_wrapper.end_det(device_index)
    return ret

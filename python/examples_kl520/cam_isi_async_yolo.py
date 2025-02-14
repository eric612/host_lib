"""
This is the example for cam ISI async YOLO.
"""
import time

from common import constants, kdp_wrapper

def handle_result(dev_idx, inf_res, r_size, frames):
    """Handle the detected results returned from the model.

    Arguments:
        dev_idx: Integer connected device ID
        inf_res: Inference result data, should be result of isi_get_result()
        r_size: Integer inference data size
        frames: List of frames captured by the video capture instance
    """
    if r_size >= 4:
        header_result = kdp_wrapper.cast_and_get(inf_res, constants.ObjectDetectionRes)
        box_result = kdp_wrapper.cast_and_get(
            header_result.boxes, constants.BoundingBox * header_result.box_count)

        boxes = []
        for box in box_result:
            boxes.append([box.x1, box.y1, box.x2, box.y2, box.score, box.class_num])
        kdp_wrapper.draw_capture_result(dev_idx, boxes, frames, "yolo")

    return 0

def user_test_cam_yolo(dev_idx, _user_id, test_loop):
    """User test cam yolo."""
    image_source_h = 480
    image_source_w = 640
    app_id = constants.AppID.APP_TINY_YOLO3
    image_size = image_source_w * image_source_h * 2
    frames = []

    # Setup video capture device.
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Start ISI mode.
    if kdp_wrapper.start_isi(dev_idx, app_id, image_source_w, image_source_h):
        return -1

    start_time = time.time()
    # Fill up the image buffers.
    ret, img_id_tx, img_left, buffer_depth = kdp_wrapper.isi_fill_buffer(
        dev_idx, capture, image_size, frames)
    if ret:
        return -1

    # Send the rest and get result in loop, with 2 images alternatively
    print("Companion image buffer depth = ", buffer_depth)
    ret = kdp_wrapper.isi_pipeline_inference(
        dev_idx, app_id, test_loop - buffer_depth, image_size,
        capture, img_id_tx, img_left, buffer_depth, frames, handle_result)

    end_time = time.time()
    diff = end_time - start_time
    estimate_runtime = float(diff / test_loop)
    fps = float(1 / estimate_runtime)
    print("Pipeline inference average estimate runtime is ", estimate_runtime)
    print("Average FPS is ", fps)

    return ret

def user_test(dev_idx, user_id):
    """User test cam ISI YOLO."""
    ret = user_test_cam_yolo(dev_idx, user_id, 1000)
    kdp_wrapper.end_det(dev_idx)
    return ret

import numpy as np
import time

from .byte_tracker import BYTETracker
from yolox.postprocess import demo_postprocess, multiclass_nms

class ByteTrackArg:
    def __init__(self, nms_thr, score_thr):
        self.score_thr = score_thr # 0.1
        self.nms_thr = nms_thr # 0.7
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.min_box_area = 100
        self.mot20 = False


def init_tracker(frame_rate, nms_thr, score_thr):
    return BYTETracker(ByteTrackArg(nms_thr, score_thr), frame_rate)


def yolox_post_process(pred, image_shape, nms_thr, score_thr):
    t0 = time.time()
    predictions = demo_postprocess(pred, image_shape, p6=False)[0]
    t1 = time.time()

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes, dtype="float32")
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    # ratio is processed in cpp side
    # boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores,
                          nms_thr=nms_thr,
                          score_thr=score_thr)
    dets = dets[:, :-1]
    t2 = time.time()
    print("post_process time p1: {:6f} p2: {:6f}".format(t1 - t0, t2 - t1))
    return dets


def update_tracker(byte_tracker, dets, img_info, image_size):
    online_targets = byte_tracker.update(dets, img_info, image_size)

    online_tlwhs = []
    online_ids = []
    online_scores = []

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > byte_tracker.args.min_box_area and not vertical:
            online_tlwhs.append(tlwh.tolist())
            online_ids.append(tid)
            online_scores.append(t.score)
    #print(online_tlwhs)
    #print(online_ids)
    #print(online_scores)
    return online_tlwhs, online_ids, online_scores


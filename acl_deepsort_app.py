import copy
import numpy as np
np.seterr(all='raise')
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class TrackerContext:

    def __init__(self):
        nn_budget = 20
        max_cosine_distance = 0.2
        self.metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)
    
    def update(self, detections):
        self.tracker.predict()
        self.tracker.update(detections)
    
    def query_tracking_result(self):
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bbox = bbox.clip(0)
            results.append([track.track_id,
                            bbox[0].item(),
                            bbox[1].item(),
                            bbox[2].item(),
                            bbox[3].item()])
        return results

def init_tracker():
    return TrackerContext()

def make_detections(boxes, scores, features):
    # boxes: input int32, convert to fp64, shape: (n, 4), format:`(x, y, w, h)`
    # scores: input float, convert to fp64, shape: (n,)
    # features: shape: (n, 128)
    boxes = copy.deepcopy(boxes)
    scores = copy.deepcopy(scores)
    features = copy.deepcopy(features)
    
    boxes = boxes.astype("float64")
    scores = scores.astype("float64")
    detection_num = len(boxes)
    print("boxes", boxes)
    print("scores", scores)

    detections = []

    for i in range(detection_num):
        detections.append(Detection(boxes[i], scores[i], features[i]))
    
    return detections

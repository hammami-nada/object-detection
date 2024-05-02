# nms stands for non_max_supression
# reverse=True means it's sorted in descending order
# bboxes :list of bounding boxes
# iou_threshold:those who are greater than or equal to this threshold will be considered redundant and suppressed
import torch
from .intersection_over_union import intersection_over_union


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        # removes the first elemnt from the list and then returned by the pop() method
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

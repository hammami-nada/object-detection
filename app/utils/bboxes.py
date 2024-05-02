import torch
from .nms import nms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bboxes(data_loader, model, iou_threshold, threshold):
    all_pred_boxes = []
    all_target_boxes = []

    for batch_idx, (x, y) in enumerate(data_loader):
        x = x.to(DEVICE)
        with torch.no_grad():
            out = model(x)

        bboxes = [[] for _ in range(len(x))]
        for i in range(len(out)):
            for j in range(2):  # Number of boxes per grid cell
                bboxes_per_img = nms(
                    out[i][j], iou_threshold=iou_threshold, threshold=threshold
                )
                bboxes[i] += bboxes_per_img

        all_pred_boxes.append(bboxes)
        all_target_boxes.append(y)

    return all_pred_boxes, all_target_boxes

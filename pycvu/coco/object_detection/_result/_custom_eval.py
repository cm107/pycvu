from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Results, BBoxResult
    from .._structs import Annotations, Annotation
from pyevu import BBox2D
import numpy as np

def _match_bounding_boxes(
    dt_boxes: list[BBox2D], gt_boxes: list[BBox2D],
    iou_threshold: float=0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    ChatGPT Assisted With Following Prompt:
    I have a list of detected bounding boxes and a list of ground truth bounding boxes.
    Write an algorithm for matching ground truth bounding boxes to detected bounding boxes based on IoU.
    Note that each ground truth bounding box can be matched to at most one detected bounding box, and vice versa.
    """
    matched_indices = []
    unmatched_dt_indices = list(range(len(dt_boxes)))
    unmatched_gt_indices = list(range(len(gt_boxes)))

    # Sort the ground truth boxes in descending order of area
    sorted_gt_boxes = sorted(
        enumerate(gt_boxes), key=lambda x: x[1].area, reverse=True
    )

    # Iterate over the sorted ground truth boxes
    for gt_index, gt_box in sorted_gt_boxes:
        best_iou = 0
        best_dt_index = None

        # Iterate over the unmatched detected boxes
        for dt_index in unmatched_dt_indices:
            dt_box = dt_boxes[dt_index]
            iou = BBox2D.IoU(dt_box, gt_box)

            # Check if the IoU is above the threshold and higher than the best IoU so far
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_dt_index = dt_index

        # Check if a match was found
        if best_dt_index is not None:
            # Add the matched indices to the list
            matched_indices.append((best_dt_index, gt_index))

            # Remove the matched indices from the unmatched lists
            unmatched_dt_indices.remove(best_dt_index)
            unmatched_gt_indices.remove(gt_index)

    return matched_indices, unmatched_dt_indices, unmatched_gt_indices

def _calculate_ap(precision, recall):
    # Append 0 at the beginning of precision and recall lists
    precision = [0] + precision
    recall = [0] + recall

    # Calculate the interpolated precision
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Calculate the area under the precision-recall curve
    # ap = np.sum((recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(precision)))
    ap = sum((recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(precision)))

    return ap

def custom_eval(
    self: Results, anns: Annotations | list[Annotation],
    iouThresh: float=0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    This doesn't quite match up with pycocotools,
    but it could still be useful for identifying tp,fp,fn.
    """
    
    imgId2anns: dict[int, list[tuple[int, Annotation]]] = {}
    imgId2results: dict[int, list[tuple[int, BBoxResult]]] = {}
    for annIdx, ann in enumerate(anns):
        if ann.image_id not in imgId2anns:
            imgId2anns[ann.image_id] = [(annIdx, ann)]
        else:
            imgId2anns[ann.image_id].append((annIdx, ann))
    for rIdx, r in enumerate(self):
        if r.image_id not in imgId2results:
            imgId2results[r.image_id] = [(rIdx, r)]
        else:
            imgId2results[r.image_id].append((rIdx, r))
    imgIds = list(set(list(imgId2anns.keys()) + list(imgId2results.keys())))
    imgIds.sort()

    imgCatSummary: dict[int, dict[int, dict]] = {}
    catImgSummary: dict[int, dict[int, dict]] = {}
    for imgId in imgIds:
        if imgId not in imgCatSummary:
            imgCatSummary[imgId] = {}
        img_anns = imgId2anns[imgId] if imgId in imgId2anns else []
        img_results = imgId2results[imgId] if imgId in imgId2results else []

        catId2anns: dict[int, list[tuple[int, Annotation]]] = {}
        catId2results: dict[int, list[tuple[int, BBoxResult]]] = {}
        for annIdx, ann in img_anns:
            if ann.category_id not in catId2anns:
                catId2anns[ann.category_id] = [(annIdx, ann)]
            else:
                catId2anns[ann.category_id].append((annIdx, ann))
        for rIdx, r in img_results:
            if r.category_id not in catId2results:
                catId2results[r.category_id] = [(rIdx, r)]
            else:
                catId2results[r.category_id].append((rIdx, r))
        
        catIds = list(set(list(catId2anns.keys()) + list(catId2results.keys())))
        catIds.sort()
        for catId in catIds:
            if catId not in catImgSummary:
                catImgSummary[catId] = {}
            _anns = catId2anns[catId] if catId in catId2anns else []
            _results = catId2results[catId] if catId in catId2results else []
            _results.sort(key=lambda x: x[1].score, reverse=True) # Does this make a difference?
            gt_boxes = [
                BBox2D(tuple(ann.bbox[:2]), (ann.bbox[0] + ann.bbox[2], ann.bbox[1] + ann.bbox[3]))
                for annIdx, ann in _anns
            ]
            dt_boxes = [
                BBox2D(tuple(r.bbox[:2]), (r.bbox[0] + r.bbox[2], r.bbox[1] + r.bbox[3]))
                for rIdx, r in _results
            ]
            matched_indices, unmatched_dt_indices, unmatched_gt_indices = _match_bounding_boxes(
                dt_boxes, gt_boxes, iou_threshold=iouThresh
            )
            tp = len(matched_indices)
            fp = len(_results) - tp
            assert fp == len(unmatched_dt_indices)
            fn = len(_anns) - tp
            assert fn == len(unmatched_gt_indices)
            # if tp + fp == 0 or tp + fn == 0:
            #     continue
            precision = tp / (tp + fp) if tp + fp != 0 else -1
            recall = tp / (tp + fn) if tp + fn != 0 else -1
            _summary = {
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tp_annIdx': [_anns[gtIdx][0] for dtIdx, gtIdx in matched_indices],
                'tp_rIdx': [_results[dtIdx][0] for dtIdx, gtIdx in matched_indices],
                'fp_rIdx': [_results[dtIdx][0] for dtIdx in unmatched_dt_indices],
                'fn_annIdx': [_anns[gtIdx][0] for gtIdx in unmatched_gt_indices]
            }
            
            # catSummary[catId] = _summary
            imgCatSummary[imgId][catId] = _summary
            catImgSummary[catId][imgId] = _summary
    
    classAp: dict[int, float] = {}
    for catId, _imgSummary in catImgSummary.items():
        _summaries = list(_imgSummary.values())
        _summaries.sort(key=lambda x: x['precision'], reverse=True) # Sort by precision in decending order

        cumulative_tp = 0
        cumulative_fp = 0
        precisions = []
        recalls = []

        total_tp = sum([_summary['tp'] for _summary in _summaries])
        total_fn = sum([_summary['fn'] for _summary in _summaries])
        for _summary in _summaries:
            precision = _summary['precision']
            if precision == -1:
                continue
            cumulative_tp += precision
            cumulative_fp += (1 - precision)
            cum_prec = cumulative_tp / (cumulative_tp + cumulative_fp)
            cum_rec = cumulative_tp / (total_tp + total_fn)

            precisions.append(cum_prec)
            recalls.append(cum_rec)
        
        ap = _calculate_ap(precisions, recalls)
        classAp[catId] = ap
    aps = list(classAp.values())
    mAP = sum(aps) / len(aps)
    return mAP, classAp, imgCatSummary

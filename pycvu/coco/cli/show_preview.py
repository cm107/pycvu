import argparse
from .. import ObjectDetection

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to coco dataset json.")
    parser.add_argument("--imgDir", type=str, help="Path to the image directory of your dataset.", default=None)
    parser.add_argument("--showBBox", action='store_true', help="Show bbox.", default=False)
    parser.add_argument("--showSeg", action='store_true', help="Show segmentation.", default=False)
    parser.add_argument("--showLabel", action='store_true', help="Show label.", default=False)
    parser.add_argument("--showAll", action='store_true', help="Show bbox, segmentation, and label.", default=False)
    return parser.parse_args()

args = get_args()
dataset = ObjectDetection.Dataset.load(args.path)
ObjectDetection.Dataset.PreviewSettings.showBBox = args.showBBox or args.showAll
ObjectDetection.Dataset.PreviewSettings.showSeg = args.showSeg or args.showAll
ObjectDetection.Dataset.PreviewSettings.showLabel = args.showLabel or args.showAll
dataset.show_preview(imgDir=args.imgDir)

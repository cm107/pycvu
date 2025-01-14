import sys
import argparse
from .. import ObjectDetection
from .. import OCR

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to coco dataset json.")
    parser.add_argument("--imgDir", type=str, help="Path to the image directory of your dataset.", default=None)
    
    parser.add_argument("--showAll", action='store_true', help="Show bbox, segmentation, and label.", default=False)

    # Object Detection Only
    parser.add_argument("--showBBox", action='store_true', help="Show bbox.", default=False)
    parser.add_argument("--showSeg", action='store_true', help="Show segmentation.", default=False)
    parser.add_argument("--showLabel", action='store_true', help="Show label.", default=False)
    
    # OCR Only
    parser.add_argument("--ocr", action='store_true', help="Assume OCR dataset.", default=False)
    parser.add_argument("--showText", action='store_true', help="Show OCR Text", default=False)
    parser.add_argument("--showQuad", action='store_true', help="Show OCR Quad", default=False)
    
    parser.add_argument("--filename", type=str, help="Path to the specific image filename that you want to see.", default=None)
    parser.add_argument("--save", type=str, help="The directory or path where you would like to save the preview images.", default=None)
    return parser.parse_args()

def main():
    args = get_args()
    if not args.ocr:
        dataset = ObjectDetection.Dataset.load(args.path)
        ObjectDetection.Dataset.PreviewSettings.showBBox = args.showBBox or args.showAll
        ObjectDetection.Dataset.PreviewSettings.showSeg = args.showSeg or args.showAll
        ObjectDetection.Dataset.PreviewSettings.showLabel = args.showLabel or args.showAll
    else:
        dataset = OCR.Dataset.load(args.path)
        OCR.Dataset.PreviewSettings.showText = args.showText or args.showAll
        OCR.Dataset.PreviewSettings.showQuad = args.showQuad or args.showAll

    if args.filename is None:
        if args.save is None:
            dataset.show_preview(imgDir=args.imgDir)
        else:
            dataset.save_preview(saveDir=args.save, imgDir=args.imgDir, showPbar=True)
    else:
        if args.save is None:
            dataset.show_filename(filename=args.filename, imgDir=args.imgDir)
        else:
            dataset.save_filename(filename=args.filename, savePath=args.save, imgDir=args.imgDir)

if __name__ == '__main__':
    sys.exit(main())
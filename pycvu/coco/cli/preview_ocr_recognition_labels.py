import sys
import argparse
from .. import OCR

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="Path to ocr recognition directory.")
    return parser.parse_args()

def main():
    args = get_args()
    OCR.Dataset.visualize_easyocr_recognition_labels(args.dirpath)

if __name__ == '__main__':
    sys.exit(main())
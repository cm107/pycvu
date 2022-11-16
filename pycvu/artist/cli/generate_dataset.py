import argparse
import os
import cv2
import numpy as np
from .. import Artist

def get_args() -> argparse.Namespace:
    def dir_path(path: str) -> str:
        if os.path.isdir(path):
            return path
        else:
            raise NotADirectoryError(path)
    
    def file_path(path: str) -> str:
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(path)
    
    def json_path(path: str) -> str:
        path = file_path(path)
        if os.path.splitext(path)[1] == '.json':
            return path
        else:
            raise ValueError(f"Must be a json file: {path}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=json_path,
        help="Path to Artist config json."
    )
    parser.add_argument(
        "--frames", type=int,
        help="Number of frames to generate per dataset.",
        default=1
    )
    parser.add_argument(
        "--repeat", type=int,
        help="Number of datasets to generate.",
        default=1
    )
    parser.add_argument(
        "--width", type=int,
        help="The width of the images to be generated.",
        default=500
    )
    parser.add_argument(
        "--height", type=int,
        help="The height of the images to be generated.",
        default=500
    )
    parser.add_argument(
        "--bg", type=file_path,
        help="Path to a background image, if any.",
        default=None
    )
    parser.add_argument(
        "--reshape_bg", action="store_true",
        help="Whether or not you want to resize the provided background to (width, height).",
        default=False
    )
    parser.add_argument(
        "--dumpDir", type=str,
        help="Directory where you want to dump the generated datasets.",
        default="artistDatasetDump"
    )
    parser.add_argument(
        "--showPbar", action="store_true",
        help="Show a progress bar.",
        default=False
    )

    return parser.parse_args()

args = get_args()

artist = Artist.load(args.config)
if args.bg is not None:
    bg = cv2.imread(args.bg)
    if args.reshape_bg:
        bg = cv2.resize(
            bg, dsize=(args.width, args.height),
            interpolation=cv2.INTER_LINEAR
        )
else:
    bg = (np.ones((args.height, args.width, 3)) * 255).astype(np.uint8)
artist.src = bg

artist.generate_dataset(
    frames=args.frames, repeat=args.repeat,
    dumpDir=args.dumpDir,
    showPbar=args.showPbar
)

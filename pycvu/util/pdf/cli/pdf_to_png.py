import argparse
# from .. import pdf_to_png
from .. import PDF

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to pdf file / folder."
    )
    parser.add_argument(
        "--output_dir",
        default="pdf_images",
        help="Path to where you want to save the converted png images."
    )
    parser.add_argument(
        "--dpi",
        default=256,
        type=int,
        help="DPI of saved image."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Whether or not you want to skip the pdfs that have already been converted."
    )
    # parser.add_argument(
    #     "--background_color",
    #     default=None,
    #     help="The color of the background you want to use if there is no background defined in the pdf."
    # )
    parser.add_argument(
        "--pool",
        default=None,
        type=int,
        help="Specify how many cpus you would like to use for multiprocessing."
    )
    parser.add_argument(
        "--save_cpus",
        default=2,
        type=int,
        help="Specify the minimum number of cpus you would like to leave available while doing multiprocessing."
    )
    parser.add_argument(
        "--showPbar",
        action="store_true",
        default=True,
        help="Show a progress bar."
    )
    return parser.parse_args()

args = get_args()
# pdf_to_png(
#     path=args.path,
#     output_dir=args.output_dir,
#     dpi=args.dpi,
#     skip_existing=args.skip_existing,
#     background_color=args.background_color,
#     pool=args.pool,
#     save_cpus=args.save_cpus,
#     showPbar=args.showPbar
# )
PDF.pdf_to_png(
    path=args.path,
    output_dir=args.output_dir,
    dpi=args.dpi,
    skip_existing=args.skip_existing,
    pool=args.pool,
    save_cpus=args.save_cpus,
    showPbar=args.showPbar
)
import os

def create_link(srcPath: str, dstPath: str):
    os.symlink(
        src=os.path.relpath(
            os.path.abspath(srcPath),
            os.path.abspath(f"{os.path.split(dstPath)[0]}")
        ),
        dst=os.path.abspath(dstPath)
    )

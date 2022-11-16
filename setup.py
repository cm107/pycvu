from setuptools import setup, find_packages
import codecs
import os.path

packages = find_packages(
        where='.',
        include=[
            'pycvu*'
        ]
)

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    # https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='pycvu',
    version=get_version("pycvu/__init__.py"),
    description='Python Computer Vision Utility',
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pylint>=2.15.5',
        'numpy>=1.23.4',
        'opencv-python>=4.6.0.66',
        'Pillow>=9.2.0',
        'PyYAML>=6.0',
        'joblib>=1.2.0',
        'pyevu @ git+ssh://git@github.com/cm107/pyevu.git@master'
    ],
    python_requires='>=3.10',
    entry_points={
        "console_scripts": [
            "pycvu-coco-show_preview=pycvu.coco.cli.show_preview"
        ]
    }
)
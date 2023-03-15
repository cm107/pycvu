from setuptools import setup, find_packages
import codecs
import os.path
from itertools import chain

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

# class SetupSpec:
#     def __init__(self, required: list[str]):
#         self.required = required
#         self.extras: dict[str, list[str]] = {}
#         self.entry_points: dict[str, list[str]] = {}
    
#     def add_console_scripts(self, value: str | list[str], extra_tags: str | list[str]=None):
#         if 'console_scripts' not in self.entry_points:
#             self.entry_points['console_scripts'] = []
#         if type(value) is str:
#             value = [value]
#         if extra_tags is not None:
#             if type(extra_tags) is str:
#                 extra_tags = [extra_tags]
#             for i in range(len(value)):
#                 value[i] = f"{value[i]} [{','.join(extra_tags)}]"
#         self.entry_points['console_scripts'].extend(value)
#         print(f"{self.entry_points['console_scripts']=}")

# setupSpec = SetupSpec(
#     [
#         'pylint>=2.15.5',
#         'numpy>=1.23.4',
#         'opencv-python>=4.6.0.66',
#         'Pillow>=9.2.0',
#         'PyYAML>=6.0',
#         'joblib>=1.2.0',
#         'tqdm>=4.64.1',
#         'pyevu @ git+ssh://git@github.com/cm107/pyevu.git@master'
#     ]
# )
# setupSpec.extras['pdf'] = [
#     'PyMuPDF==1.19.6'
# ]
# setupSpec.extras['detectron2'] = [
#     # TODO
# ]
# setupSpec.extras['nlp'] = [
#     'transformers>=4.25.1'
# ]
# setupSpec.extras['all'] = list(set(
#     chain.from_iterable(setupSpec.extras.values())
# ))

# setupSpec.add_console_scripts(
#     [
#         "pycvu-coco-show_preview=pycvu.coco.cli.show_preview",
#         "pycvu-artist-generate_dataset=pycvu.artist.cli.generate_dataset",
#     ]
# )
# setupSpec.add_console_scripts(
#     "pycvu-pdf_to_png=pycvu.util.pdf.cli.pdf_to_png",
#     extra_tags=['pdf', 'all']
# )

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
        'pylint',
        'numpy',
        'opencv-python>=4.6.0.66',
        'Pillow>=9.2.0',
        'PyYAML>=6.0',
        'joblib>=1.2.0',
        'tqdm>=4.64.1',
        'pyevu @ git+ssh://git@github.com/cm107/pyevu.git@master',
        'PyMuPDF>=1.19.6'
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            "pycvu-coco-show_preview=pycvu.coco.cli.show_preview",
            "pycvu-artist-generate_dataset=pycvu.artist.cli.generate_dataset",
        ]
    }
)
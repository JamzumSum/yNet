# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='yNet',
    version='1.0',
    description=
    'A Breast Ultrasound Image Classification Algorithm Based on Metric Learning',
    author='JamzumSum',
    author_email='zzzzss990315@gmail.com',
    url='https://github.com/JamzumSum/yNet',
    python_requires=">=3.9",
    install_requires=[
        'torch >= 1.7'
        'pytorch-lightning == 1.3.1',
        'opencv-python',
        'rich',
        'omegaconf',
        'indexserial',
    ],
    dependency_links=[
        'https://github.com/JamzumSum/IndexSerializer/releases/download/1.1.0/indexserial-1.1.0-py3-none-any.whl'
    ],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
)

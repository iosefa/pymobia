from pkg_resources import parse_requirements
from setuptools import setup, find_packages

def load_requirements(filename):
    with open(filename, 'r') as f:
        return [str(req) for req in parse_requirements(f.read())]

setup(
    name="obia",
    version="0.1.0",
    description="A Python package for object-based image analysis on georeferenced imagery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Joseph Emile Honour Percival",
    author_email="ipercival@gmail.com",
    url="https://github.com/iosefa/obia",
    license="MIT",
    packages=find_packages(exclude=["notebooks", "tests"]),
    install_requires=load_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
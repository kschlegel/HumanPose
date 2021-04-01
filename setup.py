from setuptools import setup, find_packages

setup(
    name="HumanPose",
    version="0.0.1",
    author="Kevin Schlegel",
    author_email="kevinschlegel@cantab.net",
    description="Toolbox for stuff human pose related",
    url="https://github.com/kschlegel/HumanPose",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'opencv-python', 'matplotlib'],
)

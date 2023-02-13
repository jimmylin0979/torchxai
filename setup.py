import pathlib
from setuptools import setup, find_packages

#
version = "0.0.4"

# 
with open("README.md", "r", encoding="utf-8") as fr:
    long_description = fr.read()

# 
install_requires = []
with open("requirements.txt", encoding="utf-8") as fr:
    for s in fr.readlines():
        if s[0] == "#": continue
        if s[0] == " ": continue
        install_requires.append(s.strip())

# 
# testpypi: https://test.pypi.org/project/torchxai/0.0.2/
setup(
    #
    name="torchxai",
    version=version,
    author="jimmylin0979",
    author_email="jimmylin0979@gmail.com",
    description="Visualzation methods that help developers to realize the deep network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jimmylin0979/torchxai",
    packages=find_packages(exclude=['data', 'example', 'test']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="deep learning, torch, explainable, visualization",

    #
    python_requires=">=3",
    
    # install independences, such as torch, tqdm, and so on ...
    install_requires=install_requires
)
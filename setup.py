from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fr:
    long_description = fr.read()

# testpypi: https://test.pypi.org/project/torchxai/0.0.2/
setup(
    #
    name="torchxai",
    version="0.0.2",
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

    #
    python_requires=">=3"
    
    # TODO: install independences, such as torch, tqdm, and so on ...
    # install_requires=[],

)
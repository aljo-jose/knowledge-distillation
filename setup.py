import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knowledge-distillation",
    version="0.0.1",
    author="Aljo",
    author_email="aljo@novasolution.net",
    description="knowledge distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aljo-jose/knowledge-distillation.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
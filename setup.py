import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gdf",
    version="0.0.1",
    author="Wilfried L. Bounsi",
    author_email="wilcoln99@gmail.com",
    description="A package that provides graph creation and manipulation with dataframes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wilcoln/gdf",
    packages=setuptools.find_packages(),
    install_requires=[
        'networkx',
        'cdlib',
        'pandas',
        'yake',
        'spacy',
        'scikit-learn',
        'sentence_transformers',
        'nltk',
        'keybert',
        'pyasn1',
        'google-cloud-language',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

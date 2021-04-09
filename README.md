# GDF (Graph Dataframe) Package

A package that provides keywords extraction datasets and methods

## Installation
```
pip install git+https://github.com/wilcoln/keywords.git
```
## Usage

```python
from keywords import datasets

datasets.table()  # returns dataframe of all available keywords datasets

datasets.load(<dataset_name>) # loads the dataset <dataset_name>
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://github.com/wilcoln/keywords/blob/master/LICENSE)

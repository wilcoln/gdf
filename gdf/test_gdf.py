import pandas as pd
import gdf

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='train')
data = {'text': newsgroups.data[:100], 'target': [newsgroups.target_names[target] for target in newsgroups.target[:100]]}
df = pd.DataFrame.from_dict(data)
corpus = ';'.join(list(df['text']))

print(df.graph)
print(df.graph.nb_levels())

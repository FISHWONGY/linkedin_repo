import torch
from torchtext.legacy import data, datasets
import random

seed = 966
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""**Fields**

[Check documentation](https://pytorch.org/text/_modules/torchtext/data/field.html)
"""

# define fields

"""**Text REtrieval Conference (TREC) Question Classification Dataset**

*Data Examples and Six Categories:*

| Text | Label | Category |
| --- | --- | --- |
|CNN is the abbreviation for what ?|ABBR| ABBREVIATION |
| What is the date of Boxing Day ? | NUM |NUMERIC|
|Who discovered electricity ?| HUM |HUMAN|
|What 's the colored part of the eye called ?|ENTY|ENTITY|
|Why do horseshoes bring luck ?|DESC|DESCRIPTION|
|What is California 's capital ?|LOC|LOCATION|
"""

train, test = datasets.TREC.splits(TEXT, LABEL)
train, val = train.split(random_state=random.seed(seed))

vars(train[-1])

# build vocab

print(LABEL.vocab.stoi)

print("Vocabulary size of TEXT:",len(TEXT.vocab.stoi))
print("Vocabulary size of LABEL:",len(LABEL.vocab.stoi))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, val, test),
    batch_size = 64,
    sort_key=lambda x: len(x.text), 
    device=device
)


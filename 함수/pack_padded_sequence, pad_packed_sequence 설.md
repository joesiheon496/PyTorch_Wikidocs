[simonjisu님 github](https://gist.github.com/simonjisu/b1fd54be706fd90397c6b40aa416bb17)
```python
from torch.nn import pack_padded_sequence
pack_padded_sequence(embedded, lengths.tolist(), batch_first=True)
```
RNN 모델에서 padding까지만 학습 할 수 있도

[simonjisu님 github](https://gist.github.com/simonjisu/b1fd54be706fd90397c6b40aa416bb17)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Embedding dim: E = 12
# Batch Size: B = 2
# Tokens: T = 6
# Hidden Size: H = 20
# Vocab Size: V = 10

E,B,T,H,V = (12,1,6,20,10)
embedding_layer = nn.Embedding(V,E)
lstm = nn.LSTM(E,H,1,batch_first = True, bidirectional = False)

# 데이터 불러오기
# x:data loader에서 나온 패딩이 곧 데이터라고 하면
# 아래와 같은ㅇ 코드가 있어야 한다
# for x in Dataloaer:
# ...
# 현재 배치 = 2, 패딩된 sequence 총 길이 = 6
# 즉, 배치의 1번 데이터는 길이가 6, 2번 데이터는 길이가 3임으로
# 제일 긴 1번 데이터에 맞춰서 2번 데이터를 3개의 토큰을 패딩한다.
x = torch.LongTenser([[3,9,2,1,5,6],
                      [1,2,4,0,0,0]])
# x: B,T

# 2. 모델 내부: Forward 과정
# output= model(x)

embed = embedding_layer(x)
# embed: B,T,E

# 패딩된 문자를 패킹(패딩은 연산 안들어가도록)
packed = pack_padded_sequence(embed, inputs_lengths.tolist(), batch_first=True)
# packed: B*T,E
init_hidden = torch.zeros(B,T,H)
init_cell = torch.zeros(B,T,H)
  
```
RNN 모델에서 padding까지만 학습 할 수 있도

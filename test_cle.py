import numpy as np
from dependency_decoding import chu_liu_edmonds

np.random.seed(43)
score_arc = np.random.rand(2, 3, 3)
score_root = np.random.rand(2, 3)
lengths = np.array([3, 2], dtype=np.int32)
heads = chu_liu_edmonds(score_arc, score_root, lengths)
# print(score_arc)
print(heads)

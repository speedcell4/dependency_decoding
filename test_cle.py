import numpy as np
from dependency_decoding import chu_liu_edmonds

np.random.seed(43)
score_arc = np.random.rand(2, 3, 3)
score_root = np.random.rand(2, 3)
lengths = np.array([3, 2], dtype=np.int32)
heads, scores = chu_liu_edmonds(score_arc, score_root, lengths)
# print(score_arc)
print(f'heads => {heads}')
print(f'scores => {scores}')

score_arc = np.array([[[1, 1], [2, 1]], [[1, 2], [1, 1]]], dtype=np.float64)
score_arc = np.log(score_arc)
score_root = np.array([[3, 2], [1, 2]], dtype=np.float64)
score_root = np.log(score_root)
lengths = np.array([2, 2], dtype=np.int32)
heads, scores = chu_liu_edmonds(score_arc, score_root, lengths)
scores = np.exp(np.array(scores))
# print(score_arc)
print(f'heads => {heads}')
print(f'scores => {scores}')

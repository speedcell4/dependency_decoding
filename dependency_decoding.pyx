# distutils: language = c++
# distutils: sources = decoding.cpp
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from libc.math cimport isnan

cdef extern from "decoding.h":
    cdef void c_chu_liu_edmonds(
            vector[cbool] *disabled,
            vector[vector[int]] *candidate_heads,
            vector[vector[double]] *candidate_scores,
            vector[int] *heads);


def chu_liu_edmonds(double[:,:] score_arc, double [:] score_root):
    # The size of the sentence includes the root at index 0
    cdef size_t sentence_len = len(score_arc)+1
    cdef vector[vector[int]] candidate_heads
    cdef vector[vector[double]] candidate_scores
    cdef vector[int] heads = vector[int](sentence_len, -1)
    cdef vector[cbool] disabled = vector[cbool](sentence_len, <cbool> False)

    candidate_scores.resize(sentence_len)
    candidate_heads.resize(sentence_len)

    assert score_arc.shape[0] == score_arc.shape[1], "Score matrix must be square"

    cdef int dep_i, head_i
    cdef double edge_score
    for dep_i in range(score_arc.shape[0]):

        edge_score = score_root[dep_i]
        if not isnan(edge_score):
            candidate_heads[dep_i+1].push_back(0)
            candidate_scores[dep_i+1].push_back(edge_score)

        for head_i in range(score_arc.shape[1]):
            edge_score = score_arc[dep_i, head_i]
            if not isnan(edge_score):
                candidate_heads[dep_i+1].push_back(head_i+1)
                candidate_scores[dep_i+1].push_back(edge_score)


    c_chu_liu_edmonds(disabled=&disabled, candidate_heads=&candidate_heads, candidate_scores=&candidate_scores,
                    heads=&heads)

    # Convert heads format
    return heads

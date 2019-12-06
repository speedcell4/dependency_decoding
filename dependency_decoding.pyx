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
            vector[int] *heads, double *value);

    cdef void batch_c_chu_liu_edmonds(
            vector[vector[cbool]] *disabled,
            vector[vector[vector[int]]] *candidate_heads,
            vector[vector[vector[double]]] *candidate_scores,
            vector[vector[int]] *heads,
            vector[double] *values);


def chu_liu_edmonds(double[:,:,:] score_arc, double [:,:] score_root, int [:] lengths):
    # The size of the sentence includes the root at index 0
    # cdef size_t sentence_len = len(score_arc)+1
    cdef vector[vector[vector[int]]] candidate_heads
    cdef vector[vector[vector[double]]] candidate_scores
    cdef vector[vector[int]] heads
    cdef vector[vector[cbool]] disabled
    cdef vector[double] values
    # cdef vector[int] heads = vector[int](sentence_len, -1)
    # cdef vector[cbool] disabled = vector[cbool](sentence_len, <cbool> False)

    cdef int dep_i, head_i
    cdef double edge_score
    cdef int batch_size, sentence_len, max_len

    batch_size = len(lengths)
    max_len = score_arc.shape[1] + 1
    candidate_heads.resize(batch_size)
    candidate_scores.resize(batch_size)
    heads.resize(batch_size)
    disabled.resize(batch_size)
    values.resize(batch_size)

    assert score_arc.shape[1] == score_arc.shape[2], "Score matrix must be square"

    for i, l in enumerate(lengths):
        sentence_len = l + 1
        candidate_scores[i].resize(sentence_len)
        candidate_heads[i].resize(sentence_len)
        heads[i].resize(max_len, -1)
        disabled[i].resize(sentence_len, <cbool> False)

        for dep_i in range(l):
            for head_i in range(l):
                edge_score = score_arc[i][dep_i, head_i]
                if not isnan(edge_score):
                    candidate_heads[i][dep_i+1].push_back(head_i+1)
                    candidate_scores[i][dep_i+1].push_back(edge_score)

            edge_score = score_root[i][dep_i]
            if not isnan(edge_score):
                candidate_heads[i][dep_i+1].push_back(0)
                candidate_scores[i][dep_i+1].push_back(edge_score)

    batch_c_chu_liu_edmonds(
        disabled=&disabled,
        candidate_heads=&candidate_heads,
        candidate_scores=&candidate_scores,
        heads=&heads,
        values=&values,
    )

    # Convert heads format
    return heads, values

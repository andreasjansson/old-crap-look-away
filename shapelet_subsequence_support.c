PyObject *cands;
int ncands;
PyObject *candidate;
PyObject *cls;
PyObject *example;

for(int outer = 0; outer < len_seq_candidates; outer ++) {
  for(int inner = 0; inner < len_seq_candidates; inner ++) {
    for(int i = 0; i < seq_len; i ++) {
      if(candidate_matrix[outer * seq_len + i] !=
         candidate_matrix[inner * seq_len + i])
        goto next_iteration;
    }

    cands = PySequence_GetItem(candidates, inner);
    ncands = PySequence_Size(cands);
    for(int i = 0; i < ncands; i ++) {
      candidate = PySequence_GetItem(cands, i);
      example = PyObject_GetAttrString(candidate, "example");
      subsequence_support[outer * total_examples + PyInt_AsLong(example)] += 1;
      Py_DECREF(example);
    }
    Py_DECREF(cands);

next_iteration:
    ;
  }
}

#include <math.h>

PyObject *cand_prob;

for(int t = 0; t < seq_len; t ++) {
  for(int i = 0; i < cands_len; i ++) {

    if(t + cand_seq_lens[i] >= seq_len)
      continue;

    for(int j = 0; j < cand_seq_lens[i]; j ++) {
      if(t + j < seq_len && seq[t + j] != cand_seqs[i][j])
        goto next;
    }

    cand_prob = PySequence_GetItem(cand_probs, i);

    for(int j = 0; j < nclasses; j ++) {
      class_prob[j] += *(double *)PyArray_GETPTR1(cand_prob, j);
    }

    Py_DECREF(cand_prob);

next: 
    ;
  }
}

PyObject *seq;
int seqlen;
PyObject *cand;
int candlen;

int len_seqs = PySequence_Size(seqs);
int len_cands = PySequence_Size(cands);

for(int s = 0; s < len_seqs; s ++) {

  seq = PySequence_GetItem(seqs, s);
  seqlen = PySequence_Size(seq);

  for(int c = 0; c < len_cands; c ++) {

    cand = PySequence_GetItem(cands, c);
    candlen = PySequence_Size(cand);

    for(int i = 0; i < seqlen; i ++) {

      if(i >= seqlen - candlen)
        break;

      for(int j = 0; j < candlen; j ++) {
        if(*(int *)PyArray_GETPTR1(seq, i + j) != *(int *)PyArray_GETPTR1(cand, j)) {
          goto next;
        }
      }

      occurrences[s * len_cands + c] ++;
    
next:
      ;
    }

    Py_DECREF(cand);
  }

  Py_DECREF(seq);
}

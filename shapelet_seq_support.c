PyObject *candidate;
PyObject *item;
long s;

for(int t = 0; t < seq_len; t ++) {
  for(int i = 0; i < cands_len; i ++) {

    if(t + cand_seq_lens[i] >= seq_len)
      continue;

    candidate = PySequence_GetItem(cands, i);

    for(int j = 0; j < cand_seq_lens[i]; j ++) {
      item = PySequence_GetItem(candidate, j);
      s = PyInt_AsLong(item);
      Py_DECREF(item);

      if((int)(seq[t + j]) != s) {
        goto next;
      }
    }

    seq_support[i] += 1;

next: 
    Py_DECREF(candidate);
  }
}

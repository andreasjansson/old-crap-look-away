PyObject *classes;
int nclasses;
PyObject *cls;

for(int outer = 0; outer < len_seq_candidates; outer ++) {
  for(int inner = 0; inner < len_seq_candidates; inner ++) {
    for(int i = 0; i < seq_len; i ++) {
      if(masked_candidate_matrix[outer * seq_len + i] !=
         masked_candidate_matrix[inner * seq_len + i])
        goto next_iteration;
    }

    classes = PySequence_GetItem(candidate_classes, inner);
    nclasses = PySequence_Size(classes);
    for(int i = 0; i < nclasses; i ++) {
      cls = PySequence_GetItem(classes, i);
      class_matrix[outer * total_classes + PyInt_AsLong(cls)] += 1;
      Py_DECREF(cls);
    }
    Py_DECREF(classes);

next_iteration:
    ;
  }
}

for(int outer = 0; outer < len_seq_candidates; outer ++) {
  for(int inner = 0; inner < len_seq_candidates; inner ++) {
    for(int i = 0; i < seq_len; i ++) {
      if(masked_candidate_matrix(outer, i) != masked_candidate_matrix(inner, i))
        goto next_iteration;
    }

    for(int i = 0; i < candidate_classes[inner].length(); i ++)
    //class_matrix(outer, (candidate_classes[inner])(i)) += 1;


next_iteration:
    ;
  }
}

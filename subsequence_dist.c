float best_dist = INFINITY;
int end = seq_len - subseq_len;
int sub_end;
float dist; 

for(int start = 0; start < end; start ++) {
  dist = 0;
  for(int i = 0; i < subseq_len; i ++) {
    dist += pow(seq[i + start] - subseq[i], 2);
    if(dist > best_dist)
      goto next;
  }

  if(dist < best_dist)
    best_dist = dist;
 next:
  ;
}

best_dist = sqrt(best_dist);

return_val = best_dist;

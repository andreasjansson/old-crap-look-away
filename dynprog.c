int t;
int x;
int i;
long min_cost;
int min_i;
long cost;
long a;
long weighted_cost;
long zero_weight;

for(t = 1; t < length; t ++) {

  a = amp(t);
  weighted_cost = a * change_cost;
  zero_weight = a * silence_cost;

  for(x = 0; x < height; x ++) {
    min_cost = 1000000000;
    for(i = 0; i < height; i ++) {

      if(x == 0) {
        cost = costs(i, t - 1) + zero_weight - values(x, t);
      }
      else {
        cost = costs(i, t - 1) + values(x, t);
      }

      if(i != x) {
        cost += weighted_cost;
      }

      if(cost < min_cost) {
        min_cost = cost;
        min_i = i;
      }

    }
    costs(x, t) = min_cost;
    prev(x, t) = min_i;
  }
}

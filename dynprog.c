#include <stdio.h>

int t;
int x;
int i;
int min_cost;
int min_i;
int cost;

// completely arbitrary, use some moving average maybe
int max_value = 120;

for(t = 1; t < length; t ++) {
  for(x = 0; x < height; x ++) {
    min_cost = 1000000;
    for(i = 0; i < height; i ++) {

      if(x == 0) {
        cost = costs(i, t - 1) + max_value - values(x, t);
      }
      else {
        cost = costs(i, t - 1) + values(x, t);
      }

      if(i != x)
        cost += change_cost;

      if(cost < min_cost) {
        min_cost = cost;
        min_i = i;
      }
    }
    costs(x, t) = min_cost;
    prev(x, t) = min_i;
  }
}

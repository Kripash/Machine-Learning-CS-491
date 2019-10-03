   print(max_depth)
   for x in range(features_list.shape[0]):
     if(features_list[x][0] == -1 and features_list[x][1] == -1):
       print(x, " ", features_list[x], end = ' ')
       num_00 = 0
       num_01 = 0
       num_10 = 0
       num_11 = 0
       for y in range(labels.shape[0]):
         if(features[y][x] == 0 and labels[y] == 0):
           num_00 = num_00 + 1
         if(features [y][x] == 0 and labels[y] == 1):
           num_01 = num_01 + 1
         if(features[y][x] == 1 and labels[y] == 0):
           num_10 = num_10 + 1
         if (features[y][x] == 1 and labels[y] == 1):
           num_11 = num_11 + 1
       print(num_00, num_01)
       if(num_00 + num_01 > 0):
         n_entropy = calc_entry(num_00, num_01, (num_00 + num_01))
         print(n_entropy, end = ' ')
       if (num_10 + num_11 > 0):
         y_entropy = calc_entry(num_10, num_11, (num_10 + num_11))
         print(y_entropy, end = ' ')
     print()
   #return entropy_subtree(features, labels, max_depth - 1, copy.copy(prev_entropy), copy.copy(features_list))
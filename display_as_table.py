import pandas
import numpy as np
teams_list = ["Man Utd", "Man City", "T Hotspur"]
data = np.array([[1, 2, 1],
                 [0, 1, 0],
                 [2, 4, 2]])

print (pandas.DataFrame(data, teams_list, teams_list))

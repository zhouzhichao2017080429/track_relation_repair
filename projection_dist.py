import numpy as np

class projection_dist:
    def __init__(self):
        print("")

    def distance(self,truth_T,predicted_T):
        min_dist_list = np.zeros((len(truth_T), 1))
        for i in range(len(truth_T)):
            min_dist = 1e6
            for k in range(len(predicted_T)):
                dist = np.sqrt(sum((truth_T[i,:]-predicted_T[k,:])**2))
                if dist < min_dist:
                    min_dist = dist

            min_dist_list[i, 0] = min_dist
        dist = np.sum(min_dist_list[:]) / len(truth_T)
        return dist


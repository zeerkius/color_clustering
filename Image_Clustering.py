from read_values import catch_arr
import numpy as np
import numba
import itertools
import time
import sys
import threading


k = catch_arr()

pics = np.array(k[0]) 
paths = np.array(k[1])

class csort:
    def __init__(self,data ,path):
        self.data = data
        self.path = path
        self()

    def preprocessing(self , reshape = (72,72)): 
        import numpy as np
        import os
        import cv2

        if reshape[0] != reshape[1]:
            raise ValueError("picture size must be square")
        if reshape[0] > 100:
            raise ValueError("max size exceeded")

        n_arr = np.array([])
        for img_path in os.listdir(self.path):
            new_img = cv2.imread(img_path , cv2.IMREAD_COLOR)
            new_image = np.array(cv2.resize(new_img , reshape))
            final_norm = self.normalize_array(new_image) 
            n_arr.append(final_norm)
        return n_arr

    def normalize_array(self,arr , new_min = 0 , new_max = 1):
        # since the values of the array are from {0 - 255} we want to range the values from {0 - 1}

        old_min = np.min(arr)
        old_max = np.max(arr) 

        normalized_array = new_min + (arr - old_min) * (new_max - new_min) / (old_max - old_min) # {RGB values scaled to [0 , 1]}

        return normalized_array

    def normalize_each(self): # if data will be used in a CNN / avoid integer overflow
        return self.normalize_array(self.data)
  
    def get_r15552(self):
        # so the data basically has the dimensions 72 X 72 X 3
        # in linear algbrea we know that R^(72 X 72 X 3) is a eucleidan space because it has  vector addition
        # scalar multiplication , and the inner product 
        # this means R^ (72 X 72 X 3) == R ^ 15552
        # so we will esentially go through each array and completley flatten so that we can perform the eucldiean distance
        # on the matrix point in space
        import numpy as np
        res = []
        for arr in self.data:
            res.append(arr.flatten()) # completley flattens the array
        res = np.array(res)
        return res

    def eucleidan_distance(self,arr1,arr2):
        distance  = 0
        if len(arr1) == len(arr2):
            for i in range(len(arr1)):
                distance += (arr1[i] - arr2[i]) ** 2
            distance = self.sqrt(distance)
            return distance
        else:
            print(str(len(arr1)) + "    " + str(len(arr2)))
            raise ValueError("Incompatible shape")



        
    def compute_length(self):
        return 72 * 72 * 3

    def Cmean(self,arr):
        # x = [sum(col) / len(col) for col in zip(*arr)]
        # return x
        # relying on pythons for loops for list comprehsnion is way to computationaly heavy
        # use numpy intead
        # C optimized under the hood
        import numpy as np
        means = np.mean(arr , axis = 0)
        return means

    def compute_error(self,arr1,arr2):
        import numpy as np
        a = np.sum(arr1)
        b = np.sum(arr2)
        error = (a - b) ** 2
        return error
       
    def print_error(self,arr1,arr2):
        import numpy as np
        a = np.sum(arr1)
        b = np.sum(arr2)
        error = (a - b) ** 2
        print(" Current Error ~ " +str(error))
        print('                     ')
        return error

    @staticmethod
    @numba.njit
    def sqrt(x):
        # binary search O(log(n)) takes to long for values 0 < x < 1
        # using newtons method O(log(log(n)) time 
        # fast convergence
        # f(m) = m^2 - x = 0
        # m_new = m + (x / m) / 2
        if x < 1:
            m = x
        else:
            m = x / 2
            for i in range(100):
                m = m + (x / m)
                m = m / 2
            return m

    def recreate(self,n):
        import collections
        adj = collections.defaultdict(list)
        for val in range(n):
            adj[val].append([])
            adj[val].clear()
        return adj

    def empty(self,arr):
        arr = []
        return arr

    def nsted_list(self,arr):
        res = []
        for ar in arr:
            res.append(ar.tolist())
        return res


    def fit(self , k = 4):
        import random
        import collections

        if k > 20:
            raise ValueError("max cluster limit is 20")

        vector_space = self.get_r15552() 
        path_space = self.path 

        random_centroids = [] # initialized random centroids
        new_centroids = [] # new centroids to check as they converge the optimum points

        for val in range(k):
            random_centroids.append(self.normalize_array([random.randint(0,255) for x in range(self.compute_length())]))    


        adj = collections.defaultdict(list) 

        for val in range(k):
            adj[val].append([])
 

        for n in adj:
            adj[n].clear()


        def compute_new(random_centroids , new_centroids , adj , vector_space):
            for point in vector_space:
                distance = []
                for centroid in random_centroids:
                    distance.append(self.eucleidan_distance(point,centroid))
                smallest_distance = min(distance)
                group_index = distance.index(smallest_distance)
                adj[group_index].append(point)

            new_centroids.clear()

            for cluster_id in adj:
                grouping = adj[cluster_id]
                new = self.Cmean(grouping)
                new_centroids.append(new)


            return new_centroids

        # preliminary computation

        self.print_error(random_centroids,new_centroids)

        new = compute_new(random_centroids , new_centroids , adj , vector_space)

        adj = self.recreate(k)
        new_centroids = self.empty(new_centroids)

        self.print_error(random_centroids,new)

        random_centroids = new

        newer = compute_new(random_centroids,new_centroids,adj,vector_space)

        adj = self.recreate(k)
        new_centroids = self.empty(new_centroids)

        self.print_error(random_centroids,newer)

        random_centroids = newer

        for i in range(10**18):
            newest = compute_new(random_centroids,new_centroids,adj,vector_space)
            if self.print_error(random_centroids,newest) <= 0:
                ## we will then unpack and create a file with subfolders {subfolder} in adj
                import os
                import shutil
                import collections
                adj_paths = collections.defaultdict(list)
                csort_pics = ["csort_sorted"]
                for subfolder in adj:
                    sub = str(subfolder)
                    os.makedirs(os.path.join(csort_pics[0],sub) , exist_ok=True) # makes the folder
                    string_folder = os.path.join(csort_pics[0],sub)
                    folder_values = adj[subfolder]
                    for pic in folder_values:
                        # change vector_space type
                        vp = self.nsted_list(vector_space)
                        i = vp.index(list(tuple(pic))) # to allow for comparison
                        og_path = self.path[i]
                        end = os.path.basename(self.path[i]) # end of path ie the file then its extension
                        dst = os.path.join(string_folder,end)
                        shutil.copy(og_path,dst)
                        adj_paths[subfolder].append(og_path)
                return adj_paths
            else:
                adj = self.recreate(k)
                new_centroids = self.empty(new_centroids)
                random_centroids = newest

          
    def   __call__(self):
        import numpy as np
        if type(self.data) != type(np.array([])):
            raise ValueError("Invalid DataType")
        else:
            print(' Starting Clustering ')
            print('                     ')
            self.data = self.normalize_array(self.data) # - > avoids integer overflow

 
k = csort(data= pics, path=paths) # this is instantion


def animation(stop_event):
    ani_blck = [" -" * x + "%" for x in range(15)] # we will create a loading screen 
    start = 0 # so we have to iterate so we need to have a starting point
    while not stop_event.is_set(): # so while the threading object isnt done executing
        sys.stdout.write("\r" + ani_blck[start % len(ani_blck)]) 
        sys.stdout.flush() # we need to manually call flush because write in only flushes when the buffer is full
        start += 1
        time.sleep(0.3) # to avoid the cursor just sweeping across the buffer to get the animation
        # to show properly we need to get the CPU to be idle and sleep


# now we will define the entry pointy

if __name__ == "__main__":
    stop_event = threading.Event() # - > so this is an event for each what would be thread to execture the animation
    animation_thread = threading.Thread(target=animation,args=(stop_event,)) # the comma signifies tuples
    animation_thread.start() # start thread
    k.fit(k = 9) # run clutsering
    stop_event.set() # so this allows for the little process of animation to animate , then stop
    animation_thread.join()
    print(" Clustering Finished !")




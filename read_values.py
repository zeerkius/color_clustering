

def catch_arr():
    ## this is for testing in development
    import numpy
    import os
    import cv2

    res = []
    res_paths = []

    tree = ["cats","dogs"]

    folder = r"C:\Users\agboo\RGB_Clustering Algo\cats_dogs_clustering\train"

    tuple_size = (72,72)
    for i in range(len(tree)):
        fold = os.path.join(folder,tree[i])
        for val in os.listdir(fold):
            full = os.path.join(fold,val)
            res_paths.append(full)
            img = cv2.imread(full,cv2.IMREAD_COLOR)
            new_img = numpy.array(cv2.resize(img,(72,72)))
            res.append(new_img)
    return [res,res_paths]

        








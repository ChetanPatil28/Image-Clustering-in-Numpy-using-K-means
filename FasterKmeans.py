import numpy as np
import cv2
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--num_clusters",type=int,default=5)
ap.add_argument("-iters","--iterations",type=int,default=10)
ap.add_argument('-img','--name',type=str,default='2007_000346.jpg')
args = vars(ap.parse_args())

def gray_clusters(img,num=3,num_iters=5):
    h,w=img.shape[:2]
    size=img.itemsize
    clust_img=img.copy()
    #image=img.copy()
    k=np.random.randint(0,256,size=(1,num))
    for iters in range(num_iters):
        img_view=stride_tricks.as_strided(img,shape=(h,w,1,num),strides=(w*size,1*size,0*size,0*size))
        print('Cluster Centroids',k)
        total=np.abs(img_view-k)
        total=total[:,:,0,:]
        res=np.argmin(total,axis=-1).astype(np.uint8)
        for i in range(k.shape[-1]):
            clust_img[res==i]=k[0][i]
            cords=img[res==i]
            if len(cords)==0:
                continue
            centroid=np.mean(cords)
            k[0][i]=centroid
        cv2.imshow('clustering_stages',clust_img)
        cv2.waitKey(300)
    return clust_img


if __name__ == '__main__':
    img=cv2.imread(os.path.join(os.getcwd(),args['name']),0)
    res_img=gray_clusters(img,num=args['num_clusters'],num_iters=args['iterations'])
    #cv2.imshow('Final_Clustered_Image',res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
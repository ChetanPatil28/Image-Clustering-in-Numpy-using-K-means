import numpy as np
import cv2
import matplotlib.pyplot as plt

def clust(image,k=5,iters=3): # expects img in grayscale
    img=image.copy()
    h,w=img.shape
    orig=image.copy()
    Klusters=np.random.randint(0,255,size=k)
    print('init clusters', Klusters)
    for it in range(iters):
        img=image.copy()
        for i in range(h):
            for j in range(w):
                pnt=img[i][j]
                diff=np.abs(Klusters-pnt)
                c=np.argmin(diff)
                img[i][j]=Klusters[c]
        loss=0
        l=[]
        for i in range(k):
            Ys,Xs=np.where(img==Klusters[i])
            kth_points=orig[Ys,Xs]
            l.append(np.sum(Klusters[i]-kth_points))
            Klusters[i]=np.mean(kth_points)
        loss=sum(l)    
        print('Cluster centroids at iteration-{}'.format(it+1), Klusters)
        print('loss at iteration-{}'.format(it+1),loss)
    return img


if __name__ == '__main__':
	image=cv2.imread('dog.14.jpg',0)
	clusters=clust(image,k=3)
	#print(np.unique(a))
	cv2.imshow('original_image',image)
	cv2.imshow('clustered_image',clusters)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

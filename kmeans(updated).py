import numpy as np
import cv2
import matplotlib.pyplot as plt

def clust_gray(image,k=5,iters=3): # expects img in grayscale
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
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

def clust_rgb(image,k=5,iters=3): # expects img in rgb
    img=image.copy()
    h,w,c=img.shape
    orig=image.copy()
    Klusters=np.random.randint(0,255,size=(k,3))
    print('init clusters', Klusters)
    for it in range(iters):
        img=image.copy()
        for i in range(h):
            for j in range(w):
                pnt=img[i][j]
                diff=np.sqrt(np.sum((Klusters-pnt)**2,axis=1))
                c=np.argmin(diff)
                img[i][j]=Klusters[c]
        loss=0
        l=[]
        for i in range(k):
            Ys,Xs,c=np.where(img==Klusters[i])
            kth_points=orig[Ys,Xs]
            l.append(np.sum(Klusters[i]-kth_points))
            Klusters[i]=np.mean(kth_points,axis=0)
        loss=sum(l)    
        print('Cluster centroids at iteration-{}'.format(it+1), Klusters)
        print('loss at iteration-{}'.format(it+1),loss)
    return img



if __name__ == '__main__':
	image=cv2.imread('dog.14.jpg')
	clusters=clust_rgb(image,k=3)
	cv2.imshow('original_image',image)
	cv2.imshow('clustered_image',clusters)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sp
import copy as cp


def svd(A): # singular value decomposition
    U, s, V= np.linalg.svd(A, full_matrices=True, compute_uv=True)
    sigma=sp.linalg.diagsvd(s,len(A[:,0]),len(A[0,:])) # diagonal matrix with singular values
    
    return (U,sigma,V)

def quality(n,U,S,V):  #compress image 
    sigma_low=cp.copy(S)
    for j in range(n,len(S[:,0])):  
        sigma_low[j][j]=0; # only keep the first n non-zero element while the rest are set to zeros  
    A_low=np.asmatrix(U)*np.asmatrix(sigma_low)*np.asmatrix(V) # merge back the decomposition
    A_low = np.uint8(np.array(A_low))
    
    return A_low
    
img=mpimg.imread('image.jpg')# Read image.jpg
[r,g,b] = [img[:,:,i] for i in range(3)] #split the image to red, green and blue


fig = plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(img)
ax2.imshow(r, cmap = 'Reds')
ax3.imshow(g, cmap = 'Greens')
ax4.imshow(b, cmap = 'Blues')
plt.show()


# original image decomposition

U_red, S_red, V_red =svd(r) #decomposition for red matix
U_green, S_green, V_green =svd(g) #decomposition for green matrix
U_blue, S_blue, V_blue =svd(b) # decomposition for blue matrix

#count the number of non-zero element in sigma RED, GREEN and BLUE matrix
nonzero_red=np.count_nonzero(S_red)
nonzero_green=np.count_nonzero(S_green)
nonzero_blue=np.count_nonzero(S_blue)


#compress - lower resolution picture  n=30
red_low=quality(30,U_red,S_red,V_red)  #compress red matrix
green_low=quality(30,U_green,S_green,V_green) # compress green matrix
blue_low=quality(30,U_blue,S_blue,V_blue) #compress blue matrix

rgb_low = np.dstack((red_low,green_low,blue_low)) #combining the rgb channels
fig = plt.figure(2)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(rgb_low)
ax2.imshow(red_low, cmap = 'Reds')
ax3.imshow(green_low, cmap = 'Greens')
ax4.imshow(blue_low, cmap = 'Blues')
plt.show()
fig.savefig('blur_plots.jpg')
low_im=Image.fromarray(rgb_low,'RGB')

low_im.save('rgb_low.jpg') #saving the lower resolution image as rgb_low.jpg


### compress - better resolution picture n=200

red_better=quality(200,U_red,S_red,V_red)  # compress red matrix
green_better=quality(200,U_green,S_green,V_green)  #compress green matrix
blue_better=quality(200,U_blue,S_blue,V_blue)  # compress blue matrix

rgb_better = np.dstack((red_better,green_better,blue_better)) #combining the RED, GREEN, BLUE 
fig = plt.figure(3)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(rgb_better)
ax2.imshow(red_better, cmap = 'Reds')
ax3.imshow(green_better, cmap = 'Greens')
ax4.imshow(blue_better, cmap = 'Blues')
plt.show()
fig.savefig('clear_plots.jpg')
better_im=Image.fromarray(rgb_better,'RGB')
better_im.save('rgb_better.jpg') #saving the better resolution image as rgb_better.jpg

print("The number of non-zero element in original sigma RED , GREEN , BLUE matrices are", nonzero_red,",",nonzero_green,"&",nonzero_blue,"respectively.")
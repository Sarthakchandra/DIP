from cv2 import cv2 as cv

import numpy as np
from math import *
import os

#================================================================================================

## Question 3 Solution

#================================================================================================

# # we start by taking all our inputs 
# print("Q3 Solution goes here: ")
# loc = input("Enter your image location: ")
# # loc = r"./SarthakChandra_InputImage1.jpg"
# inter = 0.5
# # loc = r"C:\Users\sarth\Desktop\x5.bmp"
# # inter = 4
# img = cv.imread(loc,0)
# print(img.shape)
# cv.imshow('input image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# #Now that we have displayed our input image, we take the interpolation factor

# print("Interpolation Factor is: ", inter)

# #We have made a function to perform billienar interpolation

# def bilinear_interpolation(img,inter):
#     a = np.array(img)
#     S = a.shape

#     # print(S)

#     R = floor(S[0]*inter)
#     C = floor(S[1]*inter)

#     # print(type(R),type(C))

#     mat = np.ones((R,C))*-1

#     #Placing the values at their specific positions

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             mat[floor(i*inter)][floor(j*inter)] = a[i][j]

#     #Range of known values

#     x_mat = floor(i*inter)
#     y_mat = floor(j*inter)
    
#     # Applying the algorithm for finding the 4 nearest neighbours as taught in the tutorial 

#     for i in range(x_mat+1):  
#         for j in range(y_mat+1):
#             if mat[i][j] == -1:

#                 #Applying bilinear Interpolation for every NN in orignal Matrix

#                 x = i/inter
#                 y = j/inter

#                 if ceil(x) != x:
#                     x1 = floor(x)
#                     x2 = ceil(x)
#                 else: 
#                     if x == 0:
#                         x1 = 0
#                         x2 = 1
#                     else:
#                         x1 = x-1
#                         x2 = x
#                 if ceil(y) != y:
#                     y1 = floor(y)
#                     y2 = ceil(y)
#                 else: 
#                     if y == 0:
#                         y1 = 0
#                         y2 = 1
#                     else:
#                         y1 = y-1
#                         y2 = y

#                 #Neighbours co-ordinate ranges

#                 x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

#                 #X Matrix for Neighbours

#                 X = [
#                     [x1,y1,x1*y1,1],[x1,y2,x1*y2,1],[x2,y2,x2*y2,1],[x2,y1,x2*y1,1]]
#                 V = [[a[x1][y1]],[a[x1][y2]],[a[x2][y2]],[a[x2][y1]]]
#                 # A = X^-1.V
#                 A = np.dot(np.linalg.inv(X),V)
#                 #Finding interpolated Pixel Value
#                 mat[i][j] = np.dot(np.array([x,y,x*y,1]),A)
    
#     # Mirroring the Borders

#     for i in range (x_mat+1):
#         for j in range(y_mat+1,len(mat[0])):
#             mat[i][j] = mat[i][j-1]

#     for j in range(len(mat[0])):
#         for i in range(x_mat+1,len(mat)):
#             mat[i][j] = mat[i-1][j]
#     return mat

# # Calling the function here

# mat_inter = bilinear_interpolation(img,inter)
# print(mat_inter.shape)
# cv.imshow("Q3_Output image",np.uint8(mat_inter))
# cv.waitKey(0)
# cv.destroyAllWindows()

#================================================================================================

## Question 4 

#================================================================================================

print("Solution to Question 4 goes here: ")

#Taking all the inputs from the user

rot = int(input("Enter Rotation Factor: "))*pi/180
scalex = float(input("Enter Scaling Factor for X axis: "))
scaley = float(input("Enter Scaling Factor for Y axis: "))
transx = int(input("Enter Translation Factor for X axis: "))
transy = int(input("Enter Translation Factor for Y axis: "))
# loc2 = r'C:\Users\sarth\Desktop\assign1.jpg'
# loc2 = r'.\SarthakChandra_Inputimage2.jpg'
loc2 = input("Enter Image Location: ")
img2 = cv.imread(loc2,0)
a = np.array(img2)
# print(len(a))

# Defining the padding for the image

pad = 7*img2.shape[0]
R_2 = img2.shape[0]
C_2 = img2.shape[1]

#displaying the input image

cv.imshow("Input",np.uint8(img2))
cv.waitKey(0)
cv.destroyAllWindows()

# print(R_2,C_2)

# Writing a function to perform Transformation in the order Rotation -> Scaling -> Translation

def transformation(rot,scalex,scaley,transx,transy):
    R = [
        [cos(rot),-sin(rot),0],
        [sin(rot),cos(rot),0],
        [0,0,1]
        ]
    R = np.array(R)
    S = [
        [scalex,0,0],
        [0,scaley,0],
        [0,0,1]
        ]
    S = np.array(S)
    T = [
        [1,0,0],
        [0,1,0],
        [transx,transy,1]
        ]
    T = np.array(T)
    T1 = np.matmul(R,S)
    Trans = np.matmul(T1,T)
    print("The Transformation Matrix is: ")
    print(Trans)
    return (Trans)

# Calling the function to perform the transformation on input parameters

mat_T = transformation(rot,scalex,scaley,transx,transy)
print(mat_T)
# Calculating the Inverse of the Transformation Matrix to be used for calculation later

mat_TI = np.linalg.inv(mat_T)
print(mat_TI)
# Writing a function for performing interpolation after transformation

def trans_interpolation(pad,R_2,C_2,mat_TI,img2,a):
    at = np.ones((pad,pad))*-1
    for i in range(pad):
        for j in range(pad):
            # print(i,j)
            try:

                # We have created a new matrix given the parameters, then we are performing 
                # matrix multiplication of the newly created matrix and mat_TI, additionally
                # We have shifted the origin for the multiplications. That is, we are 
                # essentially performing V = XT^-1

                X1 = np.matmul(np.array([i-(scalex*R_2),j-(scaley*C_2),1]),mat_TI)
                # if ((X1[0][0]>=0) and (X1[0][0] <=R_2) and (X1[0][1]>=0) and (X1[0][1] <=C_2)):
                
                #Applying bilinear Interpolation for every NN in orignal Matrix

                x = X1[0]
                y = X1[1]
                # print(x,y)
                
                if(x<0 or x >= len(a) or y < 0 or y >= len(a[0])):
                    at[i][j] = 0
                    continue
                
                if ceil(x) != x:
                    x1 = floor(x)
                    x2 = ceil(x)
                else: 
                    if x == 0:
                        x1 = 0
                        x2 = 1
                    else:
                        x1 = x-1
                        x2 = x
                if ceil(y) != y:
                    y1 = floor(y)
                    y2 = ceil(y)
                else: 
                    if y == 0:
                        y1 = 0
                        y2 = 1
                    else:
                        y1 = y-1
                        y2 = y
                
                #Neighbours co-ordinate ranges
                
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                
                #X Matrix for Neighbours
                
                X = [
                    [x1,y1,x1*y1,1],
                    [x1,y2,x1*y2,1],
                    [x2,y2,x2*y2,1],
                    [x2,y1,x2*y1,1]
                    ]
                # V Matrix for input 
                V = [
                    [img2[x1][y1]],
                    [img2[x1][y2]],
                    [img2[x2][y2]],
                    [img2[x2][y1]]
                    ]
                # A = X^-1.V
                A = np.dot(np.linalg.inv(X),V)
                #Finding interpolated Pixel Value
                at[i][j] = np.dot(np.array([x,y,x*y,1]),A)
            except:
                at[i][j] = 0
    return at
# print(at)

#Here we are calling the function for our output

mat1 = trans_interpolation(pad,R_2,C_2,mat_TI,img2,a)

# We are also using lines to indicate our Axes, to ensure that the 
# transformation can be clearly seens in our submission.

cv.line(mat1,(128,128),(128,400),(255,255,255),1)
cv.line(mat1,(128,128),(338,128),(255,255,255),1)
# cv.putText(at,"(0,0)",(38,128),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
cv.imshow("Output",np.uint8(mat1))
cv.waitKey(0)
cv.destroyAllWindows()

#================================================================================================

## Question 5 

#================================================================================================

print("Solution to Question 5 goes here: ")

# Code to export our input and output images of Q4 
# os.chdir(r"C:\Users\sarth\Desktop")
# cv.imwrite("Input.jpq",img2)
# cv.imwrite("Output.jpg",mat1)

#Using T = V-1X
# Using MS Paint to Manually find out the co-ordinates of the Matrix. The pixels co-ordinates were 
# (0,0) -> (158,158); (64,0) ->  (248,248); (64,64) -> (158,337) upto 2 decimals. Keeping in 
# mind our padding (To account for the negative values, we kept a sufficiently large padding), 
# we can allot the following values to the co-ordinate.

O = cv.imshow("Unregistered Q4 Output",np.uint8(mat1))
I = cv.imshow("Input image Q4",img2)

cv.waitKey(0)
cv.destroyAllWindows()
b = np.array(I)

# print(O[5][2],b[2])


VR = [
    [0,0,1],
    [64,0,1],
    [64,64,1]
    # [0,64,1]
    ]
XR =[
    [30,30,1],
    [120.51,-60.51,1],
    [211.02,30,1]
    # [69,247,1]
    ]

VRI = np.linalg.inv (VR)
Z = np.matmul (VRI,XR)
print(Z)
#Writing the code to register the matrix O, using Z inverse

ZI = np.linalg.inv(Z)
print(ZI)
# print(ZI)
#Calling the function we made in question 4 with parameters
Reg_O = trans_interpolation(pad,R_2,C_2,ZI,img2,a)
cv.line(Reg_O,(128,128),(128,400),(255,255,255),1)
cv.line(Reg_O,(128,128),(338,128),(255,255,255),1)
# cv.putText(at,"(0,0)",(38,128),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
cv.imshow("Registered Image",np.uint8(Reg_O))
cv.waitKey(0)
cv.destroyAllWindows()
print("We can see the images align perfectly")
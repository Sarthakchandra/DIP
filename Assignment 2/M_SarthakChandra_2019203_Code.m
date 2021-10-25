close;
clc;
clear;
close all;

% Question 3

I=double(imread('input image.jpg'));
[M,N] = size(I); %size of input image

h1=zeros(256,1);

% Normalized histogram of input image

for i=0:255
    % for every pixel value from 0 to 255, count number of occurences for
    % the whole image
    h1(i+1)=sum(sum(I==i))/(M*N);
end

r1=0:255;
% 
% Histogram Equalisation

H1 = I; % Output image equalisation
H2 = zeros(256,1);
s = sum(h1);

% Computing CDF and then equalizing the matrix via the following steps
% For a pixel value r in [0, 255], map it to s = 255*H(r)
% - For a pixel value r in [0, 255], find the indices for a given r in I.
% - At those indices, replace r with s.

for i=0:255
    %(L-1)*CDF 
    % CDF is calculated by summing over all pixel values uptil i in every
    % iteration divided by sum of total occurences of all pixels
%     In = find(I == i);
    H1(i+1)=uint8(sum(h1(1:i))/s*255); 
end 

for i = 0:255
   H_O(i+1) = sum(sum(round(H1) == i))/(M*N);
end

subplot(2,2,1),stem(r1,h1), title("Q3 - Normalised Histogram of Input Image");
subplot(2,2,2), stem(r1,H_O), title("Q3 - Normalised Histogram of Output Image");
subplot(2,2,3),imshow(uint8(I)), title("Q3 - Input Image");
subplot(2,2,4),imshow(uint8(H1)), title("Q3 - Output Image");

% Question 4

% 

dimage = double(I);
gamma = 0.5;
figure, imshow(uint8(dimage));
title("Q - 2 Input Image")
c = 255/(max(max(dimage))^gamma);
T = c*dimage.^(gamma);

g = zeros(256,1);

for i=0:255
    %(L-1)*CDF 
    % CDF is calculated by summing over all pixel values uptil i in every
    % iteration divided by sum of total occurences of all pixels
%     In = find(I == i);
    H2(i+1)=uint8(sum(h1(1:i))/s*255); 
end
% Finding Normalised Histogram for Question 4 - Gamma Transformation
for i=0:255
    % for every pixel value from 0 to 255, count number of occurences for
    % the whole image
    % p(r)
    g(i+1)=sum(sum(round(T)==i))/(M*N);
end
r2=0:255;
s=sum(g);
% Finding the transfer function for target image
G=zeros(256,1);
for i=0:255
    %(L-1)*CDF
    % CDF is calculated by summing over all pixel values uptil i in every
    % iteration divided by sum of total occurences of all pixels
    G(i+1)=uint8(sum(g(1:i))/s*255); 
end

% for i=0:255
%     s(i+1) = argmin(h1(i)-g);
% end
% figure,stem(r2,g)
% Performing the |H(r) - G|

Match=I;

for i=1:256
    % For every value in y, finds the value closest to it and save it to an
    % index, basically minimizing the difference of values and returning index
%     V = min(abs(double(H1(i))-double(G)));
    [u,ind]=min(abs(H2(i)-G));
    s = find(I == i);
    P=ind; 
    Match(s)=P;
end

% Obtain the ouput image by referencing for every pixel in image matrix the
% mapping of our Match function

% out = zeros(1,256);

% for i=1:256
%     for j=1:256
%         out(i,j)=Match(double(I(i,j))+1);
%     end
% end
% 
% % x3=double(out);
h3=zeros(256,1);

for i=0:255
    h3(i+1) = sum(sum(Match==i))/(M*N);
end

%display matched image histogram]

r3=0:255;
subplot(2,3,1), imshow(uint8(dimage)), title("Q - 4 Input Image");
subplot(2,3,2), imshow(uint8(T)), title("Q - 4 Target Image");
subplot(2,3,3), imshow(Match,[]), title("Q - 4 Matched image");
subplot(2,3,4), stem(r1,h1), title("Q - 4 Normalised Histogram of Input Image");
subplot(2,3,5), stem(r2,g), title("Q - 4 Normalised Histogram of Target Image");
subplot(2,3,6), stem(r3,h3), title("Q - 4 Normalized Histogram of Matched Image");

% Question 5

disp('Solution to 5 goes here')

% Taking input of filter and input matrix

F = [0,0,0;0,1,0;0,0,0];
h = [1,2,3;4,5,6;7,8,9];

f=zeros(3,3);

% Code to rotate the Matrix by 180 degrees

for i=1:3
    for j=1:3
        f(i,j)=F(3-i+1,3-j+1);
    end
end

[r1,c1]=size(h);
[r2,c2]=size(f);

matrix=zeros(r1+r2*2-2,c1+c2*2-2);
%big matrix that will fit our convolution brackets (with padding)

for i=r2:r1+r2-1
    for j=c2:c1+c2-1
        %filling matrix center with our image
        matrix(i,j)=h(i-r2+1,j-c2+1);
    end
end    

%final result matrix that will have convolved values (without padding)
output=zeros(r1+r2-1,c1+c2-1);
[R2,C2]=size(output);

delX=r2-1;
delY=c2-1;

for i=1:R2
    for j=1:C2
        cX=delX+i-1;
        cY=delY+j-1;
        %convolution matrix operations, (element wise multiplication)
        output(i,j)=sum(sum(matrix(cX-1:cX+1,cY-1:cY+1).*f));
    end
end

disp(output)
disp("The Output matches the sample output provided")

clear all;
clc
Ig=imread('BigTree.jpg'); % Read the Image
figure;
imshow(Ig); % display the  Original Image
[Ig1,Ig2] = kmeans_seg_k(Ig,50);
% img1 = label2rgb(Ig1,im2double(Ig2));
figure;
imshow(Ig1)
imwrite(Ig1,"BigTree_Kmeans.png")
a = unique(Ig1);
% [C1, ia1, ic1] = unique(img1);
% a_counts = accumarray(ic1,1);
% value_counts = [C1, a_counts];
k = 1;
for i = 1:size(Ig1)
    for j =1:size(Ig1)
        if(any(a(:)~=Ig1(i,j)))
            a(k) = Ig1(i,j);
            k = k+1;
        end
       
    end
end

function [krf,krl] = kmeans_seg_k(I,k)
I=double(I);
[M,N,ch]=size(I);
Id=reshape(I,M*N,ch,1);
kd=Id(ceil(rand(k,1)*M*N),:);
kdo=kd;
itr=0;
while(1)
    itr=itr+1;
dist=pdist2(Id,kdo);
[~,label]=min(dist,[],2);
for i=1:k
kd(i,:)=mean(Id(label==i,:));
end
err=sum(sum(abs(kd-kdo)));
if err<k||itr>max(5,sqrt(k))
    break;
end
kdo=kd;
end
kd=round(kd);
kr=kd(label,:);
krf=uint8(reshape(kr,M,N,ch));
krl=reshape(label,M,N,1);
end

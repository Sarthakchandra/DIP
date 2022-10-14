clc;
clear all;
close all;
% wvtool(hamming(99))
% title("H99 Filter")
% wvtool(hamming(91))
% title("H91 Filter")
% wvtool(hamming(75))
% title("H75 Filter")
% wvtool(hamming(54))
% title("H54 Filter");
A = [1 0 1 1 0;1 1 0 0 0;0 1 1 0 1;0 0 0 1 1;1 0 1 0 1; 0 1 1 1 0];
% det(A)
B = pinv(A);
% C = lsqr(A)
D = lsqminnorm(A);

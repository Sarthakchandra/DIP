clc;
clear all

syms t w g h U;
figure;
grid on;
x = -1:0.1:1;
y = dirac(x);
idx = y == Inf; % find Inf
y(idx) = 1;     % set Inf to finite value
subplot(1,2,1),stem(x,y)
title("Dirac Delta Function")
xlabel("Frequency")
ylabel("Amplitude")
g(w)=fourier(dirac(t));
h=abs(g);
w=-10:.5:10;
U=angle(g);
subplot(1,2,2),fplot(g)
title("Fourier Transform (at x = 0)")
xlabel("Frequency")
ylabel("Amplitude")
figure;
syms x

subplot(2,1,1), fplot(rectangularPulse(-1, 1, x), [-2.5 2.5]);
% axis([-1 1 0 1.2])
title({'Rectangular Pulse'})
xlabel({'Time(s)'});
ylabel('Ampltude');
syms t omega 
F(omega) = int(1*exp(1j*omega*t), t, -1, 1);
subplot(2,1,2),fplot(F, [-1 1]*16*pi)

figure;
syms x
subplot(2,1,1), fplot(sinc(x));
title("Sinc Function")
xlabel("Frequency")
ylabel("Amplitude")
z = fourier(sinc(x));
subplot(2,1,2), fplot(z);

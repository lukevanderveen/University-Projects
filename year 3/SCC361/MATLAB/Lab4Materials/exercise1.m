clear
im = imread("InfoLab.jpg");
image = imread("InfoLab.jpg");

im = im2double(im);
im = mean(im, 3);

ker = [-1, -1, -1; -1, 8, -1; -1, -1, -1];

cim = conv2(im,ker,"same");

%tmp = 0*im;
%ime = [tmp,tmp,tmp;tmp,im,tmp;tmp,tmp,tmp];
ime = [im, im, im; im, im, im; im, im, im];
cime = conv2(ime,ker,'same');
cim2 = cime(im(1)+1:2*im(1),im(2)+1:2*im(2));
mean(cim2,'all')

ulim = rot90(im,2);
uim = flipud(im);
lim = fliplr(im);
ime2 = [ulim,uim,ulim;lim,im,lim;ulim,uim,ulim];
cime2 = conv2(ime2,ker,'same');
cim22 = cime2(im(1)+1:2*im(1),im(2)+1:2*im(2));



figure(1);

subplot(1, 3, 1);
imagesc(image);
axis image off
title("oG");

subplot(1,3,2);
imagesc(cim);
axis image off
title("convolved");

subplot(1,3,3);
imagesc(abs(cime));
axis image off
title("convolved 2");

figure(2);
imagesc(ime2);
axis image off
title("convolved 3");
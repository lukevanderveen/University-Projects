clear
image = imread("LancsLogo.jpg");
im = im2double(image);
im = mean(im, 3);

im = im(275:899,:);
sz = size(im);

ker = [-1, -1, -1 ;
       -1, 8, -1;
       -1, -1, -1];

cim = conv2(im,ker,"same");

edgemap = zeros(625,1857);

for i = 1:size(im,1)
    for j = 1:size(im,2)
        if (abs(cim(i,j))) >= 0.15
            edgemap(i,j) = 1;
        end
    end
end

m = 0;
n = 0;
for i = 1:size(im,1)
    for j = 1:size(im,2)
       m = m + im(i);
       n = n + im(j);
    end
end

mean = n + m;
mean = mean*(1/(625*1857)); 
mean



figure(1)

subplot(1, 3, 1);
imagesc(image);
axis image off
title("oG");

subplot(1,3,2);
colormap("gray");
imagesc(abs(cim));
axis image off
title("convolved");

subplot(1,3,3);
imagesc(edgemap);
axis image off;
title("edged");

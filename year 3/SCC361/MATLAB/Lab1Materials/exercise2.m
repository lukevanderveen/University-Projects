image = imread("InfoLab.jpg");
x = size(image);

%grey scale
gim = mean(image, 3);
gim = gim/225;
imshow(gim);

%max pooling image
mp = zeros(floor(size(image, 1)/2), floor(size(image, 2)/2));

for i = 1:size(mp, 1)
    for j = 1:size(mp, 2)
        row_start = 2*i - 1;
        row_end = 2*i;
        col_start = 2*j -1;
        col_end = 2*j;
        
        submp = gim(row_start:row_end, col_start:col_end);

        mp(i, j) = max(submp, [], "all");
    end
end 

%displayed image 
figure(1);

subplot(1,3,1);
imagesc(image);
title("original");

subplot(1,3,2);
colormap gray;
imagesc(gim);
title("grayscale")

subplot(1, 3, 3);
imagesc(mp);
title("max pooled");

axis off:


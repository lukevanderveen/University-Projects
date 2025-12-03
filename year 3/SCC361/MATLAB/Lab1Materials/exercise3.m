image = imread("InfoLab.jpg");
image = double(image);
image = image /225;
x = imresize(image,[256, 256]);

blured = zeros(256, 256, 3);

for i = 3:size(blured, 1) -2
    for j = 3:size(blured, 2) -2
        for k = 1:size(blured, 3)

            b = x(i-2:i+2, j-2:j+2, k);

            blured(i, j, k) = mean(b, "all");
        end
    end
end

ker = ones(3,3) /9;

convolution = zeros (256, 256, 3);

for k = 1:3
    convolution = conv2(image(:,:,k), ker, "same");
end


figure(1);

subplot(1, 4, 1);
imagesc(image);
title("original");

subplot(1, 4, 2);
imagesc(x);
title("resized");

subplot(1, 4, 3);
imagesc(blured);
title("blured image");

subplot(1, 4, 4);
imagesc(blured);
title("conv2 image");
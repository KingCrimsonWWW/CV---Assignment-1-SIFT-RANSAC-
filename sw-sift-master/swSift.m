% 
%   Copyright (C) 2016  Starsky Wong <sununs11@gmail.com>
% 
%   Note: The SIFT algorithm is patented in the United States and cannot be
%   used in commercial products without a license from the University of
%   British Columbia.  For more information, refer to the file LICENSE
%   that accompanied this distribution.

clear
tic
% img1 = imread('H and F/homography_2.png');
% img2 = imread('H and F/homography_1.png');

img1 = imread('H and F/fundamental_4.png');
img2 = imread('H and F/fundamental_3.png');
% img1 = imread('VGG_Oxford_boat01.jpg');
% img2 = imread('VGG_Oxford_boat03.jpg');
% img1 = imread('VGG_Oxford_graf01.jpg');
% img2 = imread('VGG_Oxford_graf03.jpg');
% img1 = imread('book.pgm');
% img2 = imread('scene.pgm');
[des1,loc1] = getFeatures(img1);
[des2,loc2] = getFeatures(img2);
matched = match(des1,des2);
drawFeatures(img1,loc1);
drawFeatures(img2,loc2);
% print(1, '-djpeg', 'VGG_Oxford_graf01(1).jpg');
% print(2, '-djpeg', 'VGG_Oxford_graf03(1).jpg');
[num,p_i_primeraImatge,p_i_segonaImatge]=drawMatched(matched,img1,img2,loc1,loc2);
% print(3, '-djpeg', 'VGG_Oxford(F).jpg');

% %% NORANSAC
% % H=MyFindHomography(p_i_segonaImatge,p_i_primeraImatge,'affine');
% H=MyFindHomography(p_i_primeraImatge,p_i_segonaImatge,'affine');
% A = transpose(H);  %Your matrix in here
% t = maketform( 'affine', A);
% B = imtransform(img2,t);
% C=appendimages(img2,B);
% 
% figure('Name','Noransac');
% imshow(B);
% figure('Name','imag2');
% imshow(img2);
% figure('Name','imag1');
% imshow(img1);

% %% RANSAC
% % HR=MyFindHomographyRANSAC(p_i_segonaImatge,p_i_primeraImatge,'affine');
% % HR=MyFindHomographyRANSAC(p_i_primeraImatge,p_i_segonaImatge,'projective');
% % D = transpose(HR);
% % e= maketform('projective',D);
% % e= affine2d(D);
% F=imtransform(img2,t);
% 
% % G=appendimages(img1,F);
% figure('Name','RANSAC');
% imshow(F);
% 
% sizeI1 = size(img1);
% sizeI2 = size(img2);
% sizeI3 = size(F);
% %compute corner location after tansforming
% corner1 = A * [0;0;1];
% corner2 = A * [sizeI2(2);0;1];
% corner3 = A * [0;sizeI2(1);1];
% corner4 = A * [sizeI2(2);sizeI2(1);1];
% corner1 = corner1/corner1(3);
% corner2 = corner2/corner2(3);
% corner3 = corner3/corner3(3);
% corner4 = corner4/corner4(3);
% x3 = [corner1(1);corner2(1);corner3(1);corner4(1)];
% y3 = [corner1(2);corner2(2);corner3(2);corner4(2)];
% %compute offset
% offset.x = min(x3);
% offset.y = min(y3);
% %translate based on offset
% res2 = imtranslate(F,[offset.x,offset.y], 'OutputView', 'full');
% res3 = imtranslate(img1,[-min(offset.x,0),-min(offset.y,0)], 'OutputView', 'full');
% 
% % concatenate the images
% figure(2);
% res4 = my_imfuse(res2,res3);
% imshow(res4)
% figure
% % clear the dark part
% im_x = sum(res4,[2,3]);
% im_y = sum(res4,[1,3]);
% new_x = find(im_x > 0);
% new_y = find(im_y > 0);
% res5 = res4(new_x(1):new_x(end),new_y(1):new_y(end),:);
% imshow(res5);
% imwrite(res5,'result.jpg');
% % H=MyStitch(img2,t,img1);
% % figure('Name','Stitching');
% % imshow(H);
% toc
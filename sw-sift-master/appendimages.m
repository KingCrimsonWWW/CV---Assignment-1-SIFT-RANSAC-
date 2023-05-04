function im = appendimages(image1, image2)
% im = appendimages(image1, image2)
%
% Return a new image that appends the two images side-by-side.

% Select the image with the fewest rows and fill in enough empty rows
%   to make it the same height as the other image.
rows1 = size(image1,1);
rows2 = size(image2,1);
images1=image1;
images2=image2;
if (rows1 < rows2)
     images1(rows2,1) = 0;
else
     images2(rows1,1) = 0;
end

% Now append both images side-by-side.
im = [images1 images2];
end

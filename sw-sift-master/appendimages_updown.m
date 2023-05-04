function im = appendimages_updown(image1, image2)
% im = appendimages(image1, image2)
%
% Return a new image that appends the two images side-by-side.

% Select the image with the fewest rows and fill in enough empty rows
%   to make it the same height as the other image.
locs1 = size(image1,2);
locs2 = size(image2,2);
images1=image1;
images2=image2;
if (locs1 < locs2)
     images1(locs2,2) = 0;
else
     images2(locs1,2) = 0;
end

% Now append both images side-by-side.
im = [images1 ;
     images2];
end
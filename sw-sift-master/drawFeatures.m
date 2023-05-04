% 
%   Copyright (C) 2016  Starsky Wong <sununs11@gmail.com>
% 
%   Note: The SIFT algorithm is patented in the United States and cannot be
%   used in commercial products without a license from the University of
%   British Columbia.  For more information, refer to the file LICENSE
%   that accompanied this distribution.

function [] = drawFeatures( img, loc )
% Function: Draw sift feature points
figure;
imshow(img);
hold on;

% plot(loc(:,2),loc(:,1),'o');
for i=1:length(loc)
    rectangle('Position',[loc(i,2)-loc(i,4) loc(i,1)-loc(i,4) 2*loc(i,4) 2*loc(i,4)],'Curvature',[1 1])
    plot([loc(i,2),loc(i,2)+loc(i,4)*sin(loc(i,3))],[loc(i,1),loc(i,1)+loc(i,4)*cos(loc(i,3))],'-');
end

end
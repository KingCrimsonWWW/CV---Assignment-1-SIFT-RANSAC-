% 
%   Copyright (C) 2016  Starsky Wong <sununs11@gmail.com>
% 
%   Note: The SIFT algorithm is patented in the United States and cannot be
%   used in commercial products without a license from the University of
%   British Columbia.  For more information, refer to the file LICENSE
%   that accompanied this distribution.

function [ descrs, locs ] = getFeatures( input_img )
% Function: Get sift features and descriptors
global gauss_pyr;%高斯金字塔
global dog_pyr;%高斯差分金字塔
global init_sigma;%初始高斯核标准差
global octvs;%金字塔层数
global intvls;%待提取的图象数
global ddata_array;%存储特征点数据的结构体数组
global features;%存储特征点的结构体数组
if(size(input_img,3)==3)
    input_img = rgb2gray(input_img);
end
input_img = im2double(input_img);

%% Build DoG Pyramid
% initial sigma
init_sigma = 1.6;
% number of intervals per octave
intvls = 3;
s = intvls;
k = 2^(1/s);
sigma = ones(1,s+3);
sigma(1) = init_sigma;
sigma(2) = init_sigma*sqrt(k*k-1);
for i = 3:s+3
    sigma(i) = sigma(i-1)*k;
end
% default cubic method
input_img = imresize(input_img,2);
% assume the original image has a blur of sigma = 0.5
input_img = gaussian(input_img,sqrt(init_sigma^2-0.5^2*4));
%gaussian函数意为得到高斯滤波器并使用
% smallest dimension of top level is about 8 pixels
octvs = floor(log( min(size(input_img)) )/log(2) - 2);
%此为金字塔层数计算公式

% gaussian pyramid
[img_height,img_width] =  size(input_img);
gauss_pyr = cell(octvs,1);
% set image size
gimg_size = zeros(octvs,2);
gimg_size(1,:) = [img_height,img_width];
%第一层金字塔的第一幅图象是原图像
for i = 1:octvs
    if (i~=1)
        gimg_size(i,:) = [round(size(gauss_pyr{i-1},1)/2),round(size(gauss_pyr{i-1},2)/2)];
        %这里对上一层金字塔进行尺度缩小，倍数为2，得到下一层金字塔的尺度
    end
    gauss_pyr{i} = zeros( gimg_size(i,1),gimg_size(i,2),s+3 );
end
%此循环为记录金字塔每幅图像的尺度，其中gauss_pyr{i}为金字塔的层数标识
for i = 1:octvs
    for j = 1:s+3
        if (i==1 && j==1)
            gauss_pyr{i}(:,:,j) = input_img;
        % downsample for the first image in an octave, from the s+1 image
        % in previous octave.
        elseif (j==1)
            gauss_pyr{i}(:,:,j) = imresize(gauss_pyr{i-1}(:,:,s+1),0.5);
            %这里将上一层的倒数第三幅图像作为这一层的第一幅
        else
            gauss_pyr{i}(:,:,j) = gaussian(gauss_pyr{i}(:,:,j-1),sigma(j));
            %若不是第一幅，那么利用前一幅图像进行高斯滤波
        end
    end
end
%此循环为得到高斯金字塔
%% 绘出高斯金字塔
Gass_s=cell(octvs,1);
Gass_stemp=[];
Gass_o=[];
for i=1:octvs
    for j=1:s+3
    Gass_stemp=[Gass_stemp gauss_pyr{i}(:,:,j)];
    end
Gass_s{i}(:,:)=Gass_stemp;
Gass_stemp=[];
end
Gass_o=Gass_s{1}(:,:);
for i=2:octvs
    Temp=Gass_s{i}(:,:);
    Temp=padarray(Temp,[0 size(Gass_o,2)-size(Temp,2)],0,'post');

    Gass_o=[Gass_o;Temp];
end
figure('Name','高斯金字塔')
imshow(Gass_o)

%% dog pyramid高斯差分金字塔
dog_pyr = cell(octvs,1);
for i = 1:octvs
    dog_pyr{i} = zeros(gimg_size(i,1),gimg_size(i,2),s+2);
    %进行零矩阵初始化，利用存储的尺度gimg——size,而每一层图像将减少一张
    for j = 1:s+2
    dog_pyr{i}(:,:,j) = gauss_pyr{i}(:,:,j+1) - gauss_pyr{i}(:,:,j);
    %进行差分运算
    end
end
% for i = 1:size(dog_pyr,1)
%     for j = 1:size(dog_pyr{i},3)
%         imwrite(im2bw(im2uint8(dog_pyr{i}(:,:,j)),0),['dog_pyr\dog_pyr_',num2str(i),num2str(j),'.png']);
%     end
% end
%这里是将得到的高斯差分金字塔输出成文件

%% 绘出高斯差分金字塔
Dog_s=cell(octvs,1);
Dog_stemp=[];
Dog_o=[];
for i=1:octvs
    for j=1:s+2
        
        Dog_stemp=[Dog_stemp imadjust(dog_pyr{i}(:,:,j))];
    
    end
Dog_s{i}(:,:)=Dog_stemp;
Dog_stemp=[];
end
Dog_o=Dog_s{1}(:,:);
for i=2:octvs
    Temp=Dog_s{i}(:,:);
    Temp=padarray(Temp,[0 size(Dog_o,2)-size(Temp,2)],0,'post');

    Dog_o=[Dog_o;Temp];
end
figure('Name','高斯差分金字塔')
imshow(Dog_o)

%% Accurate Keypoint Localization
%即在金字塔中寻找极值点（排除有离散影响）
% width of border in which to ignore keypoints
img_border = 5;
% maximum steps of keypoint interpolation
max_interp_steps = 5;
% low threshold on feature contrast
%特征对比度阈值低
contr_thr = 0.04;
% high threshold on feature ratio of principal curvatures
%主曲率特征比阈值高
curv_thr = 3;

prelim_contr_thr = 0.5*contr_thr/intvls;
%设定阈值
ddata_array = struct('x',0,'y',0,'octv',0,'intvl',0,'x_hat',[0,0,0],'scl_octv',0);
ddata_index = 1;
for i = 1:octvs
    [height, width] = size(dog_pyr{i}(:,:,1));
    %提取每一层金字塔的尺度（height，width）
    % find extrema in middle intvls
    for j = 2:s+1
        dog_imgs = dog_pyr{i};
        dog_img = dog_imgs(:,:,j);
        %imgs指一层金字塔中的所有图像，img指第j张
        for x = img_border+1:height-img_border
            for y = img_border+1:width-img_border
                % preliminary check on contrast对比度初步检查
                if(abs(dog_img(x,y)) > prelim_contr_thr)
                    % check 26 neighboring pixels
                    if(isExtremum(j,x,y))
                        ddata = interpLocation(dog_imgs,height,width,i,j,x,y,img_border,contr_thr,max_interp_steps);
                        %解出像素点的极值点，赋予ddate
                        if(~isempty(ddata))
                            if(~isEdgeLike(dog_img,ddata.x,ddata.y,curv_thr))
                                %剔除边缘点
                                 ddata_array(ddata_index) = ddata;
                                 ddata_index = ddata_index + 1;
                            end
                        end
                    end
                end
            end
        end
    end
    %外层循环指从第二层到倒数第二层为待提取特征的层数
end
%这就是提取特征点的循环

function [ flag ] = isExtremum( intvl, x, y)
% Function: Find Extrema in 26 neighboring pixels
    value = dog_imgs(x,y,intvl);
    block = dog_imgs(x-1:x+1,y-1:y+1,intvl-1:intvl+1);
    %26个相邻像素
    if ( value > 0 && value == max(block(:)) )
        flag = 1;
    elseif ( value == min(block(:)) )
        flag = 1;
    else
        flag = 0;
    end
end

%% Orientation Assignment
%确定关键点主方向
% number of detected points
n = size(ddata_array,2);
% determines gaussian sigma for orientation assignment
ori_sig_factr = 1.5;
% number of bins in histogram
%直方图的条数
ori_hist_bins = 36;
% orientation magnitude relative to max that results in new feature
%为了增强匹配的鲁棒性，只保留峰值大于主方向峰值80％的方向作为该关键点的辅方向。
%ratio比例
ori_peak_ratio = 0.8;
% array of feature
features = struct('ddata_index',0,'x',0,'y',0,'scl',0,'ori',0,'descr',[]);
feat_index = 1;
for i = 1:n
    ddata = ddata_array(i);
    ori_sigma = ori_sig_factr * ddata.scl_octv;
    % generate a histogram for the gradient distribution around a keypoint
    hist = oriHist(gauss_pyr{ddata.octv}(:,:,ddata.intvl),ddata.x,ddata.y,ori_hist_bins,round(3*ori_sigma),ori_sigma);
    for j = 1:2
        smoothOriHist(hist,ori_hist_bins);
    end
    % generate feature from ddata and orientation hist peak
    % add orientations greater than or equal to 80% of the largest orientation magnitude
    feat_index = addOriFeatures(i,feat_index,ddata,hist,ori_hist_bins,ori_peak_ratio);
end

%% Descriptor Generation
% number of features
n = size(features,2);
% width of 2d array of orientation histograms
descr_hist_d = 4;
% bins per orientation histogram
descr_hist_obins = 8;
% threshold on magnitude of elements of descriptor vector
descr_mag_thr = 0.2;
descr_length = descr_hist_d*descr_hist_d*descr_hist_obins;
local_features = features;
local_ddata_array = ddata_array;
local_gauss_pyr = gauss_pyr;
clear features;
clear ddata_array;
clear gauss_pyr;
clear dog_pyr;
parfor feat_index = 1:n
    feat = local_features(feat_index);
    ddata = local_ddata_array(feat.ddata_index);
    gauss_img = local_gauss_pyr{ddata.octv}(:,:,ddata.intvl);
% computes the 2D array of orientation histograms that form the feature descriptor
    hist_width = 3*ddata.scl_octv;
    radius = round( hist_width * (descr_hist_d + 1) * sqrt(2) / 2 );
    feat_ori = feat.ori;
    ddata_x = ddata.x;
    ddata_y = ddata.y;
    hist = zeros(1,descr_length);
    for i = -radius:radius
        for j = -radius:radius
            j_rot = j*cos(feat_ori) - i*sin(feat_ori);
            i_rot = j*sin(feat_ori) + i*cos(feat_ori);
            r_bin = i_rot/hist_width + descr_hist_d/2 - 0.5;
            c_bin = j_rot/hist_width + descr_hist_d/2 - 0.5;
            if (r_bin > -1 && r_bin < descr_hist_d && c_bin > -1 && c_bin < descr_hist_d)
                mag_ori = calcGrad(gauss_img,ddata_x+i,ddata_y+j);
                if (mag_ori(1) ~= -1)
                    ori = mag_ori(2);
                    ori = ori - feat_ori;
                    while (ori < 0)
                        ori = ori + 2*pi;
                    end
                    % i think it's theoretically impossible
                    while (ori >= 2*pi)
                        ori = ori - 2*pi;
                        disp('###################what the fuck?###################');
                    end
                    o_bin = ori * descr_hist_obins / (2*pi);
                    w = exp( -(j_rot*j_rot+i_rot*i_rot) / (2*(0.5*descr_hist_d*hist_width)^2) );
                    hist = interpHistEntry(hist,r_bin,c_bin,o_bin,mag_ori(1)*w,descr_hist_d,descr_hist_obins);
                end
            end
        end
    end
    local_features(feat_index) = hist2Descr(feat,hist,descr_mag_thr);
end
% sort the descriptors by descending scale order
features_scl = [local_features.scl];
[~,features_order] = sort(features_scl,'descend');
% return descriptors and locations
descrs = zeros(n,descr_length);
locs = zeros(n,2);
for i = 1:n
    descrs(i,:) = local_features(features_order(i)).descr;
    locs(i,1) = local_features(features_order(i)).x;
    locs(i,2) = local_features(features_order(i)).y;
    locs(i,3) = local_features(features_order(i)).ori;
    locs(i,5) = local_features(features_order(i)).scl;
    locs(i,4)=round( 3*local_features(features_order(i)).scl * (4 + 1) * sqrt(2) / 2 );
end

end
function result = MyStitch(img_src, HRe, img_des)
%UNTITLED2 此处提供此函数的摘要
%   此处提供详细说明

[~, xdata, ydata] = imtransform(img_src,HRe);

    xdata_out=[min(1,xdata(1)) max(size(img_des,2), xdata(2))];
    ydata_out=[min(1,ydata(1)) max(size(img_des,1), ydata(2))];

    result1 = imtransform(img_src,HRe,...
        'XData',xdata_out,'YData',ydata_out);
    result2 = imtransform(img_des, maketform('affine',eye(3)),...
        'XData',xdata_out,'YData',ydata_out);
    result = result1 + result2;
    overlap = (result1 > 0.0) & (result2 > 0.0);
    result_avg = (result1/2 + result2/2); % Note overflow!
    % extra credit: Now and Then
%     result_avg = (result1);
    
    result(overlap) = result_avg(overlap);
end
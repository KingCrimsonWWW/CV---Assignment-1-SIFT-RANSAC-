function H = MyFindHomography(features,matches,model)
%UNTITLED3 此处提供此函数的摘要
%   此处提供详细说明
if strcmpi(model,'affine')==1
    A=[];
    for i=1:size(features,1)
        A=[A;features(i,1) features(i,2) 1 0 0 0;0 0 0 features(i,1) features(i,2) 1];
    end
    B=matches.';
    b=B(:);
    x=A\b;
    H=[x(1) x(2) x(3); x(4) x(5) x(6); 0 0 1];
elseif strcmpi(model,'projective')==1
    A=[];
    for i=1:size(features,1)
        A=[A;features(i,1) features(i,2) 1 0 0 0 -features(i,1)*matches(i,1) -features(i,2)*matches(i,1) ;0 0 0 features(i,1) features(i,2) 1 -features(i,1)*matches(i,2) -features(i,2)*matches(i,2)];
    end
    B=matches.';
    b=B(:);
    x=A\b;
    H=[x(1) x(2) x(3); x(4) x(5) x(6); x(7) x(8) 1];

end
end
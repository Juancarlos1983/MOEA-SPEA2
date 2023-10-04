%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YOEA122
% Project Title: Strength Pareto Evolutionary Algorithm 2 (SPEA2)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function y=Mutate2(x,params)

    h=params.h;
    VarMin=params.VarMin;
    VarMax=params.VarMax;
    nVar=numel(VarMin);      %tamanho do vetor x: 7
    sigma=h*(VarMax-VarMin);
    y=x;
    j=randsample(nVar,1); %nMu valores aleatorios entre 0-nVar:1,2,3,4,5,6,7   
    y(j)=x(j)+sigma(j)*randn(size(j));
    
    while (y(j)<VarMin(j)) | (VarMax(j)<y(j))
        y(j)=x(j)+sigma(j)*randn(size(j));
    end
end
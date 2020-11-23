function phi = EVOL_CV(I, phi0, nu, lambda_1, lambda_2, timestep, epsilon, numIter); 
%   This function updates the level set function according to the CV model  
%   input:  
%       I: input image 
%       phi0: level set function to be updated 
%       mu: weight for length term 
%       nu: weight for area term, default value 0 
%       lambda_1:  weight for c1 fitting term 
%       lambda_2:  weight for c2 fitting term 
%       muP: weight for level set regularization term  
%       timestep: time step 
%       epsilon: parameter for computing smooth Heaviside and dirac function 
%       numIter: number of iterations 
%   output:  
%       phi: updated level set function 
%   
%   created on 04/26/2004 
%   Author: Chunming Li, all right reserved 
%   email: li_chunming@hotmail.com 
%   URL:   http://www.engr.uconn.edu/~cmli/research/ 
 
phi=phi0; 
for k=1:numIter 
    phi=NeumannBoundCond(phi); 
    diracPhi=Delta(phi,epsilon); 
    Hphi=Heaviside(phi, epsilon); 
    % kappa = CURVATURE(phi,'cc');
    kappa = curvature_central(phi);
    [C1,C2]=binaryfit(I,Hphi); 
    % updating the phi function 
    phi=phi+timestep*(diracPhi.*(nu*kappa-lambda_1*(I-C1).^2+lambda_2*(I-C2).^2));     
end 


function k = curvature_central(u)                       
% compute curvature
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                       % the norm of the gradient plus a small possitive number 
                                                        % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,junk] = gradient(Nx);                              
[junk,nyy] = gradient(Ny);                              
k = nxx+nyy;                                            % compute divergence
 
 
function H = Heaviside(phi,epsilon)  
H = 0.5*(1+ (2/pi)*atan(phi./epsilon)); 
 
function Delta_h = Delta(phi, epsilon) 
Delta_h=(epsilon/pi)./(epsilon^2+ phi.^2); 
 
function g = NeumannBoundCond(f) 
% Make a function satisfy Neumann boundary condition 
[nrow,ncol] = size(f); 
g = f; 
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);   
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);           
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]); 


function K = CURVATURE(f,diff_scheme) 
% CURVATURE computes curvature 
% Author: Chunming Li, all rights reserved. 
% Email: li_chunming@hotmail.com 
% URL: http://www.engr.uconn.edu/~cmli/ 
 
epsilon=1e-10; 
 
if strcmp(diff_scheme, 'fcb')   
    [fx,fy]=gradient(f);  % central difference 
    fx_f = Dx_forward(f); % forward difference 
    ax = fx_f./sqrt(fx_f.^2+ fy.^2+epsilon); 
    axx = Dx_backward(ax); % backward difference 
    fy_f = Dy_forward(f); 
    ay = fy_f./sqrt(fx.^2 + fy_f.^2 + epsilon); 
    ayy = Dy_backward(ay);  
    K = axx + ayy; 
 
elseif strcmp(diff_scheme, 'fb')   % forward difference followed by a backward difference 
 
    fx_f = Dx_forward(f); 
    fy_f = Dy_forward(f); 
    ax = fx_f./sqrt(fx_f.^2+ fy_f.^2+epsilon); 
    ay = fy_f./sqrt(fx_f.^2 + fy_f.^2 + epsilon); 
    axx = Dx_backward(ax); 
    ayy = Dy_backward(ay); 
    K = axx + ayy; 
elseif strcmp(diff_scheme, 'bf')   % forward difference followed by a backward difference 
 
    fx_f = Dx_backward(f); 
    fy_f = Dy_backward(f); 
    ax = fx_f./sqrt(fx_f.^2+ fy_f.^2+epsilon); 
    ay = fy_f./sqrt(fx_f.^2 + fy_f.^2 + epsilon); 
    axx = Dx_forward(ax); 
    ayy = Dy_forward(ay); 
    K = axx + ayy; 
elseif strcmp(diff_scheme, 'cc')   % central difference followed by a central difference 
    [fx, fy]= gradient(f); % central difference 
    ax = fx./sqrt(fx.^2+ fy.^2+epsilon); 
    ay = fy./sqrt(fx.^2 + fy.^2 + epsilon); 
    [axx, axy] = gradient(ax); % central difference 
    [ayx, ayy] = gradient(ay); 
    K = axx + ayy;     
else 
    disp('Wrong difference scheme: CURVATURE.m'); 
    return;     
end 

function [C1,C2]= binaryfit(Img,H_phi) 
 
a= H_phi.*Img; 
numer_1=sum(a(:));  
denom_1=sum(H_phi(:)); 
C1 = numer_1/denom_1; 
 
b=(1-H_phi).*Img; 
numer_2=sum(b(:)); 
c=1-H_phi; 
denom_2=sum(c(:)); 
C2 = numer_2/denom_2; 
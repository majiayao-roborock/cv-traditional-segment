function u=LLC_MOST_NEW(u0,Img,Img_smooth,Img2_smooth,kappa,K,lambda_w,lambda_b,w_star,b_star,mu,nu,timestep)
u=u0;
u=NeumannBoundCond(u);
%divnormU=curvature_central(u);  % div(\frac{\nabla u}{|\nabla u|});
signU = tanh(kappa*u);% sgn(u)
dersignU = (kappa./(cosh(kappa*u).^2));%sgn'(u)
signU_smooth = imfilter(signU,K); %T_bar
IsignU_smooth = imfilter(signU.*Img,K);%F
tmp = 1 ./ (1+lambda_b);
w = (lambda_w.*w_star+IsignU_smooth-tmp .* Img_smooth .*(signU_smooth+lambda_b.*b_star)); 
w = w ./ ( lambda_w + Img2_smooth - tmp.* Img_smooth .*Img_smooth);
b = tmp .* ( lambda_b .* b_star + signU_smooth - w.*Img_smooth);

dE  = mu*dersignU .* (w.*Img+b-signU );
dP  = nu*(4*del2(u)); % smooth term

u = u + timestep*(dE+dP);


function g = NeumannBoundCond(f)
% Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)                       
% compute curvature
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-10);                       % the norm of the gradient plus a small possitive number 
                                                        % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,junk] = gradient(Nx);                              
[junk,nyy] = gradient(Ny);                              
k = nxx+nyy;  
function f = Dirac(x, sigma)
f=(1/((sqrt(pi)*sigma)))*exp(-x.^2/(sigma*sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;
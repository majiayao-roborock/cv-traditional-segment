function u = LLC( inputIm , initialLSF , imgID , isShow );

switch imgID
    case 1
        bord        = 1;
end


Img         = inputIm;
u           = bord * initialLSF;
timestep    = 0.005;

iterNum     = 12000;
K_sigma     = 8;
w_star      = -1;
b_star      = -mean(Img(:))*w_star;
lambda_w    = 0.0001;
lambda_b    = 1;
kappa       = 7;
nu          = 23.0;     % smooth item
mu          = 0.2;      % weighting coefficient of error function energy E


switch imgID
    case 8
        iterNum     = 6000;
        mu          = 0.02;      % weighting coefficient of error function energy E
    case 9
        iterNum     = 6000;
        mu          = 0.025;      % weighting coefficient of error function energy E
    case 10
        iterNum     = 6000;
        mu          = 0.01;      % weighting coefficient of error function energy E
    case 11
        K_sigma     = 8*3;
    case 14
        mu          = 0.025;     % weighting coefficient of error function energy E
end



K=fspecial('gaussian',round(2*K_sigma)*2+1,K_sigma);     % the Gaussian kernel
Img_smooth = imfilter(Img,K);
Img2_smooth = imfilter(Img.*Img,K);

% start level set evolution
for n=1:iterNum
    u  = LLC_MOST_NEW(u,Img,Img_smooth,Img2_smooth,kappa,K,lambda_w,lambda_b,w_star,b_star,mu,nu,timestep);
    if mod(n,20)==0 & isShow == 1
        pause(0.1);
        imagesc(inputIm, [0, 255]);colormap(gray);hold on;axis off,axis equal
        [c,h] = contour(u,[0 0],'r');
        contour(initialLSF,[0 0],'b');
        iterNum=[num2str(n), ' iterations'];
        title(iterNum);
        hold off;
    end
end
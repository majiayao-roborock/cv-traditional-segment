

function u = LBF( inputIm , initialLSF , imgID , isShow );

u           = initialLSF;
timestep    = .1;
mu          = 1;

epsilon     = 1.0;
sigma       = 3.0;

switch imgID
    case 1
        iterNum     = 2000;
        lambda1     = 1.0;
        lambda2     = 2;
        nu          = 0.004*255*255;
        bord        = 1;
end

K           = fspecial('gaussian',round(2*sigma)*2+1,sigma);

I           = inputIm;
KI          = conv2(inputIm,K,'same');
                                                 
KONE        = conv2(ones(size(inputIm)),K,'same');

% start level set evolution
for n=1:iterNum
    [u f1 f2]=RSF(u,I,K,KI,KONE, nu,timestep,mu,lambda1,lambda2,epsilon,1);
    if mod(n,20)==0 & isShow == 1
        pause(0.1);
        subplot(221);
        imagesc(inputIm, [0, 255]);colormap(gray);hold on;axis off,axis equal
        [c,h] = contour(u,[0 0],'r');
        contour(initialLSF,[0 0],'b');
        iterNumShow=[num2str(n), ' iterations'];
        title(iterNumShow);
        hold off;
        subplot(223);
        imshow(f1,[]);
        subplot(224);
        imshow(f2,[]);
    end
end


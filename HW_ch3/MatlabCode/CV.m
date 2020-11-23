

function u = CV( inputIm , initialLSF , imgID , isShow );

    u           = initialLSF; 
    numIter     = 100; 
    timestep    = 0.1; 
    epsilon     = 1; 
    switch imgID
        case 1
            lambda1     = 1.0;
            lambda2     = 2;
            nu          = 0.004*255*255;
            bord        = 1;
    end

    for k=1:numIter 
        u=EVOL_CV(inputIm, u, nu, lambda1, lambda2, timestep, epsilon, 1);   % update level set function
        
        if mod(k,20)==0 & isShow == 1
            pause(0.1);
            imagesc(inputIm, [0, 255]);colormap(gray);
            hold on;axis off,axis equal
            [c,h] = contour(u,[0 0],'r');
            contour(initialLSF,[0 0],'b');
            iterNumShow=[num2str(k), ' iterations'];
            title(iterNumShow);
            hold off;
        end
        
    end

end


function robustSegConf = RubostSeg( robustSegConf , isShow );

    curIm    = robustSegConf.param.inputIm;
    kernel   = robustSegConf.commonParam.kernel;
    lambda1  = robustSegConf.param.lambda1;
    lambda2  = robustSegConf.param.lambda2;
    nu       = robustSegConf.param.nu;
    bord     = robustSegConf.param.isBrightOrDark;
    mu       = robustSegConf.commonParam.mu;
    iterNum  = robustSegConf.param.iterNum;
    
    epsilon  = robustSegConf.commonParam.epsilon;
    timeStep = robustSegConf.commonParam.timeStep;
    
    u        = robustSegConf.param.initialLSF;
    w        = ones(size(curIm));
    
    m        = 1.00;
    
    for dIter = 1 : iterNum
        
        u       = NeumannBoundCond( u );
        k       = CurvatureCentral( u );
        
        DrcU    = (epsilon/pi)./(epsilon^2.+u.^2);
        Hu      = 0.5*(1+(2/pi)*atan(u./epsilon));
        
        [f1,f2] = LocalBinaryFit( curIm , Hu , w , kernel , bord , m );
                            
        s1          = lambda1.*f1.^2 - lambda2.*f2.^2;             
        s2          = lambda1.*f1    - lambda2.*f2;
        dataForce   = (lambda1-lambda2)*conv2(w,kernel,'same').*curIm.*curIm+conv2(s1.*w,kernel,'same')-2.*curIm.*conv2(s2.*w,kernel,'same');

        
        dist      = lambda1*(conv2(Hu.^m,kernel,'same').*curIm.*curIm-2*conv2(Hu.^m.*f1,kernel,'same').*curIm+conv2(Hu.^m.*f1.*f1,kernel,'same')) ...
                  + lambda2*(conv2((1-Hu).^m,kernel,'same').*curIm.*curIm-2*conv2((1-Hu).^m.*f2,kernel,'same').*curIm+conv2((1-Hu).^m.*f2.*f2,kernel,'same'));
        sigmaDist = 255*255/10;
        w         = exp( -dist/sigmaDist );
        w(w<0.1)  = 0;

        
        A = -Hu.^(m-1).*DrcU.*dataForce;
        P = mu*(4*del2(u)-k);
        L = nu.*DrcU.*k;
        
        u = u + timeStep*( A + L + P );
        
        if mod( dIter , 20 ) == 0 & isShow == 1
            showResult( curIm , u , f1 , f2 , w , dIter );
        end

    end
    
    robustSegConf.result.u  = u;
    robustSegConf.result.w  = w;
    robustSegConf.result.f1 = f1;
    robustSegConf.result.f2 = f2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function showResult( curIm , u , f1 , f2 , w , dIter );
    figure( 100001 );
    pause( 0.1 );
    subplot( 2 , 3 , 1 );
    imagesc( curIm, [0, 255]);
    colormap(gray);
    hold on; axis off; axis equal;
    [c,h] = contour( u , [0.1 ] , 'r' );
    title( ['这是第' num2str(dIter) '次迭代！'] );
    hold off;
    subplot( 2 , 3 , 2 );
    mesh( u );
    title( 'Final Level Set Function' );
    subplot( 2 , 3 , 3 );
    imshow( u , [] );
    title( 'Final Level Set Function' );
    subplot( 2 , 3 , 4 );
    imshow( f1 , [] );
    subplot( 2 , 3 , 5 );
    imshow( f2 , [] );
    subplot( 2 , 3 , 6 );
    imshow( w , [] );
end

function [ f1 , f2 ] = LocalBinaryFit( im , Hu , w , kernel , bord , m )

    I   = im.*Hu.^m;
    c1  = conv2( Hu.^m.*w , kernel , 'same' );                             
    c2  = conv2( I.*w  , kernel , 'same' );
    f1  = c2./(c1);
    I   = im.*(1-Hu).^m;
    c1  = conv2( (1-Hu).^m.*w , kernel , 'same' );                             
    c2  = conv2( I.*w      , kernel , 'same' );
    f2  = c2./(c1);
    if bord == 1
        f11 = min( f1 , f2 );
        f22 = max( f1 , f2 );
    elseif bord == -1
        f11 = max( f1 , f2 );
        f22 = min( f1 , f2 );
    else
        f11 = f1;
        f22 = f2;
    end
    f1  = f11;                  
    f2  = f22;
    
end

function g = NeumannBoundCond( f );

    [ h w ]          = size(f);
    g                = f;
    g([1 h],[1 w])   = g([3 h-2],[3 w-2]);  
    g([1 h],2:end-1) = g([3 h-2],2:end-1);          
    g(2:end-1,[1 w]) = g(2:end-1,[3 w-2]);
    
end

function k = CurvatureCentral( u );

    [ux,uy]     = gradient(u);                                  
    normDu      = sqrt(ux.^2+uy.^2+1e-10);
    nx          = ux./normDu;                                       
    ny          = uy./normDu;
    [nxx,junk]  = gradient(nx);                              
    [junk,nyy]  = gradient(ny);                              
    k           = nxx+nyy;
    
end


function u = CK( inputIm , initialLSF , imgID , isShow );

timestep    = .1;
mu          = 1;

epsilon     = 1.0;
sigma       = 3.0;

switch imgID
    case 1
        iterNum     = 2000;
        lambda1     = 1.0;
        lambda2     = 1.6;
        nu          = 0.004*255*255;
        bord        = 1;
end

K           = fspecial('gaussian',round(2*sigma)*2+1,sigma);


robustSegConf.commonParam.kernel    = K;
robustSegConf.param.lambda1         = lambda1;
robustSegConf.param.lambda2         = lambda2;
robustSegConf.param.nu              = nu;
robustSegConf.param.isBrightOrDark  = bord;
robustSegConf.commonParam.mu        = mu;
robustSegConf.param.iterNum         = iterNum;
robustSegConf.commonParam.epsilon   = epsilon;
robustSegConf.commonParam.timeStep  = timestep;
robustSegConf.param.initialLSF      = initialLSF;
robustSegConf.param.inputIm         = inputIm;

robustSegConf = RubostSeg(robustSegConf,isShow);

u = robustSegConf.result.u;

function [ c1 c2 ] = BinaryFit( dctInputIm , dctImOne , HPhi , weight , c1Old , c2Old , isInit );

    if isInit == 1
        %%%%    dctInputIm = inputIm
        inputIm     = dctInputIm;
        OneHPhi     = 1 - HPhi;
        c1          = sum(HPhi(:).*inputIm(:))/sum(HPhi(:));
        c2          = sum(OneHPhi(:).*inputIm(:))/sum(OneHPhi(:));
    else
        dctHPhi     = dct2( HPhi );
        dctOneHPhi  = dctImOne - dctHPhi; % dctOneHPhi  = dct2( 1 - HPhi );
        c1          = sum(weight(:).*(dctInputIm(:)-dctOneHPhi(:)*c2Old).*dctHPhi(:))/sum(weight(:).*dctHPhi(:).*dctHPhi(:));
        c2          = sum(weight(:).*(dctInputIm(:)-dctHPhi(:)*c1Old).*dctOneHPhi(:))/sum(weight(:).*dctOneHPhi(:).*dctOneHPhi(:));
    end

end
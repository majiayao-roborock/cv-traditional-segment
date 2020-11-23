

function [inputIm initialLSF] = GetImgAndInitialLSF( imgID , initContourSelectInd );

    inputIm = imread(['data/' num2str(imgID) '.bmp']);
    inputIm = double(inputIm(:,:,1));
    
    rho     = 2;
    switch imgID
        case 1
            initialLSF = ones(size(inputIm)).*rho;
            % initialLSF(20:70,20:100) = -rho;
            initialLSF(15:25,30:90) = -rho;
    end

end
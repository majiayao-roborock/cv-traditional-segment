
close all;
clear all;
clc;

methodNameList          = {'CV','LBF','CK','LLC','GO'};

for imgID = 1
    
    [inputIm, initialLSF] = GetImgAndInitialLSF( imgID , 1 );
    
    for methodInd = 1:5
        
        close all;
        switch methodNameList{1,methodInd}
            case 'CV'
                u = CV( inputIm , initialLSF , imgID , 1 );
            case 'LBF'
                u = LBF( inputIm , initialLSF , imgID , 1 );
            case 'CK'
                u = CK( inputIm , initialLSF , imgID , 1 );
            case 'LLC'
                u = LLC( inputIm , initialLSF , imgID , 1 );
            case 'GO'
                u = GO( inputIm , initialLSF , imgID , 1 );
        end
            
        close all;
        figure( 10005 );
        imshow(inputIm(:,:,1),[ ]);
        hold on;
        contour(u,[0 0],'r','linewidth',1);
        contour(initialLSF,[0 0],'b','linewidth',1);
        hold off;
        pause
        
    end
    
end


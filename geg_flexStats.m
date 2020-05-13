function geg_flexStats()
% This function computes:
% 1) Internal reliability (Chrombach's alpha) for all tests
% 2) Associations (LASSO + Correlation)
% 3) GLM interaction
% 
% For correlation analyses are several options, but rank partial
% correlations were used fo the manuscript
% 
% Author: Gabriel Gonzalez-Escamilla $ 28.08.2019_13:57, last version 12.05.2020_09:41
% 

plotRegInteraction = true;%false;%
plotLasso = true;%false;

inpath = pwd;
xlsName = 'FREEflex_dataTable.xlsx';
indTab = 'ItemData';
scoreTab = 'SummaryScores';

[numInd,txtInd,~] = xlsread(fullfile(inpath,xlsName),indTab);
[numSC,txtSC,~] = xlsread(fullfile(inpath,xlsName),scoreTab);

Age = numInd(:,10);
Gend = numInd(:,6);

FREEind = numInd(:,11:26);
FREE_PEind = numInd(:,11:14);
FREE_NEind = numInd(:,15:18);
FREE_PSind = numInd(:,19:22);
FREE_NSind = numInd(:,23:26);
FREEexp = numInd(:,11:18);
FREEsup = numInd(:,19:26);

% opath = fullfile(inpath,'regressions');% check if makes sance call to plots and then also save the PCA
% if ~exist(opath,'dir'), mkdir(opath); end


fprintf('Computing internal reliability ... \n')
% Internal reliability (Chrombach's alpha)
addpath(fullfile(inpath, 'extrafunctions'))

% FREE-scale
[stand_alphaPE,raw_alphaPE] = CronbachAlpha(FREE_PEind);% positive enhancement
[stand_alphaNE,raw_alphaNE] = CronbachAlpha(FREE_NEind);% negative enhancement
[stand_alphaPS,raw_alphaPS] = CronbachAlpha(FREE_PSind);% positive suppression
[stand_alphaNS,raw_alphaNS] = CronbachAlpha(FREE_NSind);% negative suppression
[stand_alphaE,raw_alphaE] = CronbachAlpha(FREEexp); % expressive enhancement
[stand_alphaS,raw_alphaS] = CronbachAlpha(FREEsup); % suppressive regulation
[stand_alphaF,raw_alphaF] = CronbachAlpha(FREEind);% emotional flexibility index

% RS-11
[stand_alphaRS11,raw_alphaRS11] = CronbachAlpha(numInd(:,27:37));
nRS11 = txtInd(:,27:37);% to ensure the right colums are taken

% GBB24
[stand_alphaGBB24,raw_alphaGBB24] = CronbachAlpha(numInd(:,38:47));
nGBB24 = txtInd(:,38:47);

% WHO
[stand_alphaWHO,raw_alphaWHO] = CronbachAlpha(numInd(:,48:52));
nWHO5 = txtInd(:,48:52);

% DASS
DASSd = numInd(:,53:73); nDASS = txtInd(:,53:73);
% Zur Skala D (Depression) gehören die Items 3, 5, 10, 13, 16, 17, 21.
% Die Skala A (Angst) enthält die Fragen 2, 4, 7, 9, 15, 19, 20.
% Zur Skala S (Stress) zählen die Items 1, 6, 8, 11, 12, 14, 18.
DASSD = DASSd(:,[3, 5, 10, 13, 16, 17, 21]);
DASSa = DASSd(:,[2, 4, 7, 9, 15, 19, 20]);
DASSs = DASSd(:,[1, 6, 8, 11, 12, 14, 18]);
[stand_alphaDASSd,raw_alphaDASSD] = CronbachAlpha(DASSD);% Depression
[stand_alphaDASSa,raw_alphaDASSa] = CronbachAlpha(DASSa);% Anxiety
[stand_alphaDASSs,raw_alphaDASSs] = CronbachAlpha(DASSs);% Stress

% BFI-10
BFId = numInd(:,74:83); nBFI = txtInd(:,74:83);
% Neurotizismus wird durch die Items 4R und 9 erfasst,
% Extraversion durch die Items 1 und 6,
% Offenheit durch die Items 5 und 10,
% Verträglichkeit durch die Items 2 und 7R und
% Gewissenhaftigkeit durch die Items 3R und 8.
% R = negativ gepolte Item rekodiert (Items are reversed-scored = 1, 3, 4, 5 und 7)
BFIn = BFId(:,[4, 9]); 
BFIe = BFId(:,[1, 6]); 
BFIo = BFId(:,[5, 10]); 
BFIv = BFId(:,[2, 7]); 
BFIg = BFId(:,[3, 8]); 
[stand_alphaBFI10n,raw_alphaBFI10n] = CronbachAlpha(BFIn);% Emotional stability
[stand_alphaBFI10m,raw_alphaBFI10m] = CronbachAlpha(BFIe);% Extraversion
[stand_alphaBFI10o,raw_alphaBFI10o] = CronbachAlpha(BFIo);% Openness
[stand_alphaBFI10v,raw_alphaBFI10v] = CronbachAlpha(BFIv);% Agreeableness
[stand_alphaBFI10g,raw_alphaBFI10g] = CronbachAlpha(BFIg);% Conscientiousness

% coFlex (CFS)
coFlexd = numInd(:,84:93); ncoFlex = txtInd(:,84:93);
% 1.Evaluation	2, 6, 7, 8, 9
% 2.Adaptation    1, 3, 4, 5, 10
coFlexEv = coFlexd(:,[2, 6, 7, 8, 9]);
coFlexAd = coFlexd(:,[1, 3, 4, 5, 10]);
[stand_alphacoFlexE,raw_alphacoFlexE] = CronbachAlpha(coFlexEv);% 
[stand_alphacoFlexA,raw_alphacoFlexA] = CronbachAlpha(coFlexAd);% 

% SEK
SEKd = numInd(:,94:120); nSEK = txtInd(:,94:120);
% 1.Aufmerksamkeit: Items 1, 12, 19
% 2.Klarheit: Items 6,13, 25
% 3.Körperwahrnehmung: Items 7, 14, 24
% 4.Verstehen: Items 3, 11, 20
% 5.Akzeptanz: Items 5, 17, 23
% 6.Resilienz: Items 4, 18, 26
% 7.Selbstunterstützung: Items 9, 15, 27
% 8.Konfrontationsbereitschaft: Items 8, 16, 22
% 9.Regulation: Items 2, 10, 21
% 10. ALL
SEKau = SEKd(:,[1, 12, 19]);
SEKkl = SEKd(:,[6,13, 25]);
SEKkw = SEKd(:,[7, 14, 24]);
SEKv = SEKd(:,[3, 11, 20]);
SEKak = SEKd(:,[5, 17, 23]);
SEKr = SEKd(:,[4, 18, 26]);
SEKsu = SEKd(:,[9, 15, 27]);
SEKkb = SEKd(:,[8, 16, 22]);
SEKrg = SEKd(:,[2, 10, 21]);
[stand_alphaSEKau,raw_alphaSEKau] = CronbachAlpha(SEKau);% awareness
[stand_alphaSEKkl,raw_alphaSEKkl] = CronbachAlpha(SEKkl);% clarity
[stand_alphaSEKkw,raw_alphaSEKkw] = CronbachAlpha(SEKkw);% sensations
[stand_alphaSEKkv,raw_alphaSEKv] = CronbachAlpha(SEKv);% understanding
[stand_alphaSEKak,raw_alphaSEKak] = CronbachAlpha(SEKak);% acceptance
[stand_alphaSEKr,raw_alphaSEKr] = CronbachAlpha(SEKr);% tolerance
[stand_alphaSEKsu,raw_alphaSEKsu] = CronbachAlpha(SEKsu);% self-support
[stand_alphaSEKkb,raw_alphaSEKkb] = CronbachAlpha(SEKkb);% readiness to confront 
[stand_alphaSEKrg,raw_alphaSEKrg] = CronbachAlpha(SEKrg);% modification
[stand_alphaSEK,raw_alphaSEK] = CronbachAlpha(SEKd);% Total score

% ERQ
ERQd = numInd(:,121:130); nERQ = txtInd(:,121:130);
% 1.Neubewertung (k = 6): Items 1, 3, 5, 7, 8, 10; 
% 2.Unterdrückung / Suppression (k = 4): Items 2, 4, 6, 9; 
ERQn = ERQd(:,[1, 3, 5, 7, 8, 10]);
ERQu = ERQd(:,[2, 4, 6, 9]);
[stand_alphaERQr,raw_alphaERQr] = CronbachAlpha(ERQn);% Reappraisal
[stand_alphaERQs,raw_alphaERQs] = CronbachAlpha(ERQu);% Suppression

% SWE (GSE)
[stand_alphaSWE,raw_alphaSWE] = CronbachAlpha(numInd(:,131:140));
nSWE = txtInd(:,131:140);

% % The raw_alpha* were simply copied

% % % % % % % %  
% Get summary score index data
% FREE-scale
FREE_PE = numSC(:,11);
FREE_NE = numSC(:,12);
FREE_PS = numSC(:,13);
FREE_NS = numSC(:,14);
FREE_enhanceSC = numSC(:,15);%17
FREE_supressSC = numSC(:,16);%18
FREE_flexSC = numSC(:,21);
% RS-11
RS11 = numSC(:,23);nRS11 = txtSC(1,23);
% GBB24
GBB24 = numSC(:,24); nGBB24 = txtSC(1,24);
% WHO
WHO5 = numSC(:,25); nWHO5 = txtSC(1,25);
% DASS
DASSD = numSC(:,27); nDASSD = txtSC(1,27);%-9
DASSa = numSC(:,28); nDASSa = txtSC(1,28);
DASSs = numSC(:,29); nDASSs = txtSC(1,29);
% BFI-10
BFIn = numSC(:,40); nBFIn = txtSC(1,31);
BFIe = numSC(:,39); nBFIe = txtSC(1,30);
BFIo = numSC(:,41); nBFIo = txtSC(1,32);
BFIv = numSC(:,42); nBFIv = txtSC(1,33);
BFIg = numSC(:,43); nBFIg = txtSC(1,34);
% coFlex
coFlexEv = numSC(:,35); ncoFlexEv = txtSC(1,35);
coFlexAd = numSC(:,36); ncoFlexAd = txtSC(1,36);
% SEK
SEKau = numSC(:,37); nSEKau = txtSC(1,37); 
SEKkl = numSC(:,38); nSEKkl = txtSC(1,38);
SEKkw = numSC(:,39); nSEKkw = txtSC(1,39); 
SEKv = numSC(:,40); nSEKv = txtSC(1,40); 
SEKak = numSC(:,41); nSEKak = txtSC(1,41); 
SEKr = numSC(:,42); nSEKr = txtSC(1,42); 
SEKsu = numSC(:,43); nSEKsu = txtSC(1,43);
SEKkb = numSC(:,44); nSEKkb = txtSC(1,44); 
SEKrg = numSC(:,45); nSEKrg = txtSC(1,45); 
SEK = numSC(:,46); nSEK = txtSC(1,46); 
% ERQ
ERQrea = numSC(:,47); nERQrea = txtSC(1,47); 
ERQsup = numSC(:,48); nERQsup = txtSC(1,48);
% SWE
SWE = numSC(:,49); nSWE = txtSC(1,49);


fprintf('Modelling associations ... \n')
% associations between FREE and oher variables
indV = [FREE_enhanceSC,FREE_supressSC,FREE_flexSC];
indN = [txtSC(1,15),txtSC(1,16),txtSC(1,21)];
% 
VarOfInt = [RS11,GBB24,WHO5,DASSD,DASSa,DASSs,BFIn,BFIe,BFIo,BFIv,BFIg,coFlexEv,coFlexAd,SEKau,SEKkl,SEKkw,SEKv,SEKak,SEKr,SEKsu,SEKkb,SEKrg,SEK,ERQrea,ERQsup,SWE];
VarOfIntN = [nRS11,nGBB24,nWHO5,nDASSD,nDASSa,nDASSs,nBFIn,nBFIe,nBFIo,nBFIv,nBFIg,ncoFlexEv,ncoFlexAd,nSEKau,nSEKkl,nSEKkw,nSEKv,nSEKak,nSEKr,nSEKsu,nSEKkb,nSEKrg,nSEK, nERQrea,nERQsup,nSWE];
GPnuiV1 = [Age,Gend];
% 
% % LASSO variable selection
fprintf('Selecting variables of interest ...\n')
% 
% Remove Redundant Predictors by Using Cross-Validated Fits
% https://de.mathworks.com/help/stats/lasso.html
% https://de.mathworks.com/help/stats/lasso-regularization.html %only plot1
% Construct the lasso fit by using 10-fold cross-validation with labeled predictor variables.
rng default % For reproducibility 
for i = 1:3
    [B,FitInfo] = lasso(VarOfInt,indV(:,i),'CV',10,'PredictorNames',cellstr(VarOfIntN));
    % Find the variables in the model that corresponds to the minimum cross-validated mean squared error (MSE).
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0);
    % Find the variables in the sparsest model within one standard error of the minimum MSE.
    idxLambda1SE = FitInfo.Index1SE;
    sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0);
    % Compute the FMSE of each model returned by lasso.
    yFLasso = FitInfo.Intercept + VarOfInt*B;  fmseLasso = sqrt(mean((indV(:,i) - yFLasso).^2,1));
    if plotLasso
        FitInfo2plot = rmfield(FitInfo,{'SE','LambdaMinMSE','Lambda1SE','IndexMinMSE','Index1SE'}) ;
        hax = lassoPlot(B,FitInfo2plot); L1Vals = hax.Children.XData; yyaxis right; 
        h = plot(L1Vals,fmseLasso,'LineWidth',2,'LineStyle','--'); legend(h,'FMSE','Location','NW'); ylabel('FMSE'); title('CV10 Lasso')
        % 
        % Plot the model result and cross-validated fits.
        lassoPlot(B,FitInfo,'PlotType','CV');
        legend('show') % Show legend
        % The green circle and dotted line locate the Lambda with minimum cross-validation error.
        % The blue circle and dotted line locate the point with minimum cross-validation error plus one standard deviation.
    end
    fmsebestlasso = min(fmseLasso(FitInfo.DF == 22));idx = fmseLasso == fmsebestlasso; 
    bestLasso1 = [FitInfo.Intercept(idx); B(:,idx)]; fprintf('best Lasso FMSE: %s\n',num2str(fmsebestlasso))
    lam = FitInfo.Index1SE;% Find the Lambda value of the minimal cross-validated mean squared error plus one standard deviation.
    B(:,lam);
    rhat = VarOfInt\indV(:,i);% find the least-squares estimate
    res = VarOfInt*rhat - indV(:,i); % Calculate residuals
    fprintf('fit FMSE: %s\n',num2str(FitInfo.MSE(lam)))
    MSEmin = res'*res/200; % B(:,lam) MSE value is higher, and rhat hast less zeros so rhat is the best model
    fprintf('fit least-squares FMSE: %s\n',num2str(MSEmin))
    posits = rhat~=0; isSig.(indN{i}) = VarOfIntN(posits);% define associated regions for each response variable
    bestLasso = bestLasso1(2:end); table(rhat,bestLasso,'RowNames',cellstr(VarOfIntN))
end
close all

% % % % % % % % % % % % % Correls % % % % % % % % % % % 
% Re-select the subfields to be only the ones of interest (showing network
% changes)
fprintf('Computing correlations ... \n')

corrtype = 'partialRank';%'pearson';%'rank';%'robust';%'partial';%
p1tailed = false;

warning off
for i = 1:size(indV,2)
    GP1ind = indV(:,i);
    xlabelN = indN{:,i};
    %
    n1 = size(GP1ind,1);
    [Lia,~] = ismember(VarOfIntN,isSig.(indN{i}));
    VarOfIntNSel = VarOfIntN(1,Lia);
    VarOfIntSel = VarOfInt(:,Lia);
    
    for ii = 1:length(VarOfIntNSel)
        GP1dep = VarOfIntSel(:,ii);%
        ylabelN = VarOfIntNSel{:,ii};
        aVars = [xlabelN '-' ylabelN];
        
        % % Here perfoms the correlation
        switch corrtype
            case 'partial'
                [RHO1,PVAL1] = partialcorr(GP1ind,GP1dep,GPnuiV1);
            case 'partialRank'
                [RHO1,PVAL1] = partialcorr(GP1ind,GP1dep,GPnuiV1,'Type','Spearman');
            case 'robust'
                [b1,stats1] = robustfit(GP1ind,GP1dep);%resturns the betas
                curt1 = stats1.t(2);
                RHO1 = sign(curt1)*sqrt(curt1^2/(numel(GP1ind)-1-2+curt1^2));
                PVAL1 = 2*tcdf(curt1, numel(GP1ind)-1-2); %2 tailed
            case {'pearson','linear'}
                [RHO1,PVAL1] = corr(GP1ind,GP1dep);
            case {'spearman','rank','nonlinear'}
                [RHO1,PVAL1] = corr(GP1ind,GP1dep,'Type','Spearman');
        end
        
        % (optional) make pvalues 1-tailed
        if p1tailed
            PVAL1t = PVAL1/2;
            PVAL1 = PVAL1t;
        end
        
        if PVAL1<=0.05
            fprintf(2,'%s \t r = %s \t p = %s \n', aVars,num2str(RHO1),num2str(PVAL1));
        else
            fprintf('%s \t r = %s \t p = %s \n', aVars,num2str(RHO1),num2str(PVAL1));
        end
        
        rOUT(ii,i) = RHO1;
        pOUT(ii,i) = PVAL1;
        
    end
    p2eval = pOUT(:,i);
    r2sig = rOUT(:,i);
    [~,~,padj] = fdr(p2eval');% Get FDR-adjusted p-values
    FDRsig3 = padj<0.05;
    pFDRsig3 = padj(FDRsig3)';
    rFDRsig3 = r2sig(FDRsig3);
    FDRsig3N = VarOfIntNSel(FDRsig3)';
    
    fprintf('\n')
end
warning on

fprintf('Modelling GLM interactions ... \n')
% % The flexiblity concept proposes that the measures will show stronger
% relationship to adjustment when moderated by adversity/stress.
% Therefore, as suggested by George Bonanno, now we test two things:
% -> the interaction of enhancement and suppression ability
% -> the interaction of flexiblity and the stressor/adversity predicting adjustment (e.g. depression, anxiety and stress) 
intable = table(FREE_enhanceSC,FREE_supressSC,FREE_flexSC,Age,Gend,...
    RS11,GBB24,WHO5,DASSD,DASSa,DASSs,BFIn,BFIe,BFIo,BFIv,BFIg,coFlexEv,coFlexAd,SEKau,SEKkl,SEKkw,SEKv,SEKak,SEKr,SEKsu,SEKkb,SEKrg,SEK,ERQrea,ERQsup,SWE);
% % Run three times: one for each DASS scale
for i = 1:3
    if i ==1
        fprintf(2,'\n\nComputing interaction model for DASS-depression.\n');
        mlfit = fitlm(intable,'DASSD ~ 1 + Age + Gend + FREE_enhanceSC:FREE_supressSC');
    elseif i ==2
        fprintf(2,'\n\nComputing interaction model for DASS-anxiety.\n');
        mlfit = fitlm(intable,'DASSa ~ 1 + Age + Gend + FREE_enhanceSC:FREE_supressSC');
    else
        fprintf(2,'\n\nComputing interaction model for DASS-stress.\n');
        mlfit = fitlm(intable,'DASSs ~ 1 + Age + Gend + FREE_enhanceSC:FREE_supressSC');
    end
    %
    disp(mlfit)
    % The printed table has the statistics to report:
    % Each column (Estimate, SE, tStat and pValue) are computed for the
    % covariates and the interaction. These values i the interaction are
    % the result of the interaction coefficients after accounting for the
    % main effects of the two covariates (Age and Gender). SO, are the most
    % interesting statistics from this table.
    % Under the table are the main stats for the whole model (f-value and associated against the null model)
    %
    if plotRegInteraction
        figure; plotInteraction(mlfit,'FREE_supressSC','FREE_enhanceSC')
        %figure; plotInteraction(mlfit,'FREE_supressSC','FREE_enhanceSC','predictions')
        close all
    end
end

clear all
disp('FINISHED')

function [Cap,ReqFreq_THz,BeamWidthRx_deg_max,...
    BeamWidthTx_deg_max,SNRout_dB,Gain_dB,Rate,berkay_count] = Distance_to_Cap_Opt(...
                        PowerAllocFlag,TxPower_dBmW,...
                        CoordRx,CoordTx,...
                        StartFreqTHz,StopFreqTHz,...
                        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
                        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
                        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
                        BandWidthOptFlag,BeamWidthOptFlag,Freq_THz,berkay_count)
%% Error calculate
if length(StartFreqTHz)~=length(StopFreqTHz)
    error('length(StartFreqTHz)==length(StopFreqTHz). should satisfy')
end

%% Active frequency region
ReqIdx = [];
for ii = 1:length(StartFreqTHz)
  Idx_ii = find((Freq_THz>=StartFreqTHz(ii))&...
                        (Freq_THz<=StopFreqTHz(ii)));
  ReqIdx = [ReqIdx;Idx_ii];
end
ReqIdx_Init_Orig = unique(ReqIdx);

maskIdx_Init_Orig = zeros(size(Freq_THz));
maskIdx_Init_Orig(ReqIdx_Init_Orig) = 1;
%% Gain optimization
if BeamWidthOptFlag == 0
    BeamWidthRx_deg_max = BeamWidthRx_deg;
    BeamWidthTx_deg_max = BeamWidthTx_deg;
    SideLobeStrengthRx_max = SideLobeStrengthRx;
    SideLobeStrengthTx_max = SideLobeStrengthTx;
    ProbAli = 1;
elseif BeamWidthOptFlag == 1
    [BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
        SideLobeStrengthRx_max,SideLobeStrengthTx_max,ProbAli] = ...
        AntennaGainOptim(BeamWidthRx_deg,BeamWidthTx_deg,...
        SideLobeStrengthRx,SideLobeStrengthTx,TxPower_dBmW,...
        CoordRx,CoordTx, AEtypeRx,AEtypeTx,...
        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
        maskIdx_Init_Orig,Freq_THz);
else
    error('BeamWidthOptFlag == 0,1')
end
[AbsorptionLoss_dB_GainOpt,...
    PLtotal_dB_GainOpt,Gain_dB_GainOpt,TxPower_dBW_GainOpt] = ...
            Distance_to_Loss(TxPower_dBmW,...
            CoordRx,CoordTx,...
            AEtypeRx,BeamWidthRx_deg_max,SideLobeStrengthRx_max,...
            AEtypeTx,BeamWidthTx_deg_max,SideLobeStrengthTx_max,...
            thetaTxBoreSight_deg,phiTxBoreSight_deg,...
            thetaRxBoreSight_deg,phiRxBoreSight_deg,Freq_THz);
%%
maskIdx_Init = maskIdx_Init_Orig;
maskIdx = maskIdx_Init;
maskOut = maskIdx_Init;
distance_m = sqrt(sum((CoordRx-CoordTx).^2));
AbsorptionLoss_dB = AbsorptionLoss_dB_GainOpt;
PLtotal_dB = PLtotal_dB_GainOpt;
Gain_dB = Gain_dB_GainOpt;
TxPower_dBW = TxPower_dBW_GainOpt;
while 1
    %% Power Allocaiton
    maskOutOld = maskOut;
    if PowerAllocFlag == 0
        PowerNBProdNsub_dBW = pow2db(maskIdx_Init.*db2pow(TxPower_dBW+Gain_dB));
        maskOut = maskIdx_Init;
    elseif PowerAllocFlag == 1
        % Water Filling approach
        [PowerNBProdNsub_dBW,maskOut] = WaterFillingAlgo(maskIdx_Init,...
            AbsorptionLoss_dB,PLtotal_dB,...
            TxPower_dBW+Gain_dB,Freq_THz);
    else
        error('PowerAllocFlag == 0,1');
    end
    %% BandWidth Optimization
    ReqIdx_Init = find(maskOut==1);   
    maskIdxOld = maskIdx;
    if BandWidthOptFlag == 0
        maskIdx = maskOut;
    elseif BandWidthOptFlag == 1
        maskIdx = ActiveFreqMask(maskOut,Freq_THz,...
            PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,ReqIdx_Init);
    elseif BandWidthOptFlag == 2
        maskIdx = CommonFlatBandFreqMask(maskOut,Freq_THz,...
            PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,ReqIdx_Init,...
            distance_m);
    else
        error('BandWidthOptFlag == 0,1')
    end  
    % Stopping criteria
    A = find(maskIdx);
    B = find(maskOut);
    if isempty(setdiff(union(A,B),intersect(A,B)))
        FlagBreak = 1;
        break;
    end
    maskIdx_Init = maskIdx;
end
if FlagBreak == 1
    [Cap,ReqFreq_THz,SNRout_dB,berkay_count] = Mask_Cap(maskOut,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,berkay_count);
elseif FlagBreak == 2
    PowerNBProdNsub_W = (db2pow(PowerNBProdNsub_dBW).*maskIdx);
    PowerNBProdNsub_W = db2pow(TxPower_dBW+Gain_dB)/sum(PowerNBProdNsub_W).*PowerNBProdNsub_W;
    PowerNBProdNsub_dBW = pow2db(PowerNBProdNsub_W);
    [Cap,ReqFreq_THz,SNRout_dB,berkay_count] = Mask_Cap(maskIdx,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,berkay_count);
elseif FlagBreak == 3
    [Cap,ReqFreq_THz,SNRout_dB,berkay_count] = Mask_Cap(maskOut,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,berkay_count);
else
    error('Invalid FlagBreak, 0 1 2')
end

Gain_dB_misaligned = pow2db(SideLobeStrengthRx); + pow2db(SideLobeStrengthTx);
[Cap_misaligned, ~, ~,berkay_count] = Mask_Cap(maskIdx, Freq_THz, PLtotal_dB, AbsorptionLoss_dB, Gain_dB_misaligned, TxPower_dBW,berkay_count);
Rate = ProbAli * Cap + (1-ProbAli) * Cap_misaligned;
end
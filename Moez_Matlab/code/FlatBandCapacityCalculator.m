%% Flat Band Capacity Calculator

function [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(Distance,AbsLoss_dB,FreqTHz,ReqIdx,PLtotal_dB)

deltaFreq = DeltaFreq(FreqTHz.*1e12);
NoisePwrReq = NoisePower(AbsLoss_dB,deltaFreq);

% Path Loss Flat Regions
% AbsorptionLossInvalidIdx = find((AbsLoss_dB)>=(max(AbsLoss_dB)-(max(AbsLoss_dB)-min(AbsLoss_dB))/20));
% FlatnessCriteriaPL = Distance*0.01;%+alpha0;
FlatnessCriteriaPL = Distance*0.01+1;

% Distance in meters
% FlatnessCriteriaPL = 50;
% FlatnessCriteriaPL = 1;%0.125; %% dB scale total variation.
NumberOfPoints = 10;    %% -- Bandwidth points
[Idx_Total_PL_Flat_All] = FlatRegionIdentifier(PLtotal_dB,FlatnessCriteriaPL,NumberOfPoints);
% Noise Flat Regions
%% Akhtar
AbsorptionLossThreshold = 400;
AbsorptionLossInvalidIdx = find(AbsLoss_dB>=AbsorptionLossThreshold);
%%
C = intersect(Idx_Total_PL_Flat_All,AbsorptionLossInvalidIdx);
Idx_Total_PL_Flat = setdiff(Idx_Total_PL_Flat_All,C);

FlatnessCriteriaNoise = FlatnessCriteriaPL;%0.125; %% dBm scale total variation.
% NumberOfPoints = 20;    %% --
[Idx_NoiseFlat] = FlatRegionIdentifier(pow2db(NoisePwrReq*1e3),FlatnessCriteriaNoise,NumberOfPoints);
Idx_Total_PLpNoise_Flat = intersect(Idx_Total_PL_Flat(:),Idx_NoiseFlat(:));
Idx_Total_PLpNoise_Flat = intersect(ReqIdx(:),Idx_Total_PLpNoise_Flat(:));
end
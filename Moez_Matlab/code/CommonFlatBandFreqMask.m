%% Common Flat Band Frequency Band for Max Capacity
function maskIdx = CommonFlatBandFreqMask(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW,ReqIdx_Init,dist_m)

% [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(dist_m,...
%                             AbsorptionLoss_dB,Freq_THz,ReqIdx_Init);
[Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(dist_m,...
                            AbsorptionLoss_dB,Freq_THz,ReqIdx_Init,PLtotal_dB);

maskPL_NoiseFlat = zeros(size(maskIdx_Init),'like',maskIdx_Init);
maskPL_NoiseFlat((Idx_Total_PLpNoise_Flat)) = 1;
maskIdx = maskIdx_Init & maskPL_NoiseFlat;
end
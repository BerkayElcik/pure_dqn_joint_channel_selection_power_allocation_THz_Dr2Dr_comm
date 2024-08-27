%% Calculate masking to capacity with active frquencies
function [Cap,ReqFreq_THz,SNRout,berkay_count] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW,berkay_count)
Idx = find(maskIdx==1);
DeltaFreq_Hz = DeltaFreq(Freq_THz*1e12);
ReqFreq_THz = Freq_THz(Idx);
ReqDeltaFreq = DeltaFreq_Hz(Idx);

if length(TxPower_dBW) == 1
    ReqTxPower_dBW = TxPower_dBW;
else
    ReqTxPower_dBW = TxPower_dBW(Idx);
end
ReqPLtotal_dB = PLtotal_dB(Idx);
ReqAbsorptionLoss_dB = AbsorptionLoss_dB(Idx);
[SNRout, berkay_count] = PL_Noise2SNR(ReqPLtotal_dB,ReqAbsorptionLoss_dB,...
                        Gain_dB,ReqTxPower_dBW,ReqDeltaFreq,berkay_count);

Cap = sum(ReqDeltaFreq(:).*log2(1+SNRout(:)));
end
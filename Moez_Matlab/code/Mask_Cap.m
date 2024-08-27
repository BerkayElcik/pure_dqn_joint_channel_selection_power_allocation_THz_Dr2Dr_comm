%% Calculate masking to capacity with active frquencies
function [Cap,ReqFreq_THz,SNRout] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW)
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
SNRout = PL_Noise2SNR(ReqPLtotal_dB,ReqAbsorptionLoss_dB,...
                        Gain_dB,ReqTxPower_dBW,ReqDeltaFreq);

Cap = sum(ReqDeltaFreq(:).*log2(1+SNRout(:)));
end
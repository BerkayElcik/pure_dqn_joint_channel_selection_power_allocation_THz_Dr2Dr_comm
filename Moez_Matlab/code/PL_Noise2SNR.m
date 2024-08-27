%% PL+Noise to SNR
function [SNRout] = PL_Noise2SNR(Atotal_dB,AbsLoss_dB,Gain_dB,P_tx_dB,deltaFreq)
% Path Loss Calculations
Atotal_dB_Dummy = Atotal_dB;
Atotal_dB_Dummy(Atotal_dB_Dummy==Inf)=[];
Atotal_dB_DummyMax = max(Atotal_dB_Dummy);
Atotal_dB(find(Atotal_dB>=Atotal_dB_DummyMax)) = Atotal_dB_DummyMax;
Atotal = db2pow(Atotal_dB);
% Noise power calculations
Pnoise = NoisePower(AbsLoss_dB,deltaFreq);
% Equal Power Allocation
Gain = db2pow(Gain_dB);
P_tx = db2pow(P_tx_dB);
P_Total = P_tx(:).*Gain(:);
N_subBands = length(deltaFreq);
TransmitPower = P_Total/N_subBands;
% SNR calculations
SNRout = TransmitPower(:)./(Atotal(:).*Pnoise(:));
end
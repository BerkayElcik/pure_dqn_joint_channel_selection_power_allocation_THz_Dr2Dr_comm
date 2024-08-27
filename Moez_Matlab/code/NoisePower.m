%% Noise Power calcualtions
function Pnoise = NoisePower(AbsLoss_dB,deltaFreq)
if length(AbsLoss_dB)~=length(deltaFreq)
    error('Check the dimentions of AbsLoss_dB, deltaFreq')
end
Trans = 1./db2pow(AbsLoss_dB);
Trans(find(Trans>=1)) = 0.9999999999;
Epsilon = 1 - Trans;
T0 = 296;
Tmol = T0 * Epsilon;
Tnoise = Tmol;
kB = 1.3806e-23;
Pnoise = kB.*Tnoise.*deltaFreq;
end
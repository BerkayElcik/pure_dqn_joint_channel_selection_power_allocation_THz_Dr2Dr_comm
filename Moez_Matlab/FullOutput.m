function [Trans,AbsCoeff,AbsorptionLoss_dB,PLtotal_dB,SR] = ...
    FullOutput(Freq_THz,h1,h2,d,theta)
% Transmittance
Freq_Hz = Freq_THz.*1e12;
AbsLoss_dB = AtmAbsorption(Freq_THz, h1, theta);
Trans = 1./db2pow(AbsLoss_dB);
SR = sqrt(d^2 + (h2 - h1)^2 - 2 * d * (h2 - h1) * cosd(theta));
% Absorption Coefficients K(f)
minTransVal = min(Trans(Trans~=0))*1e-10;
Trans(Trans==0) = minTransVal;
AbsCoeff = (-log(Trans))./SR;
% Total Absorption Loss [dB]
AbsorptionLoss_dB = ...
    AbsCoeff.*SR.*10.*log10(2.71828);
% Total Path Loss [dB]
PLspread_dB = 20.*log10(SR) + ...
    20.*log10(Freq_Hz./1e6)-27.55;%-GtdB-GrdB;
PLabs_dB = AbsorptionLoss_dB;
PLtotal_dB = PLspread_dB + PLabs_dB;
end
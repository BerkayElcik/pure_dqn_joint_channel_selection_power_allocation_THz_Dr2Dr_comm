function [AbsorptionLoss_dB,PLtotal_dB,Gain_dB,TxPower_dBW] = ...
                        Distance_to_Loss(TxPower_dBmW,...
                        CoordRx,CoordTx,...
                        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
                        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
                        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                        thetaRxBoreSight_deg,phiRxBoreSight_deg,Freq_THz)

%% Derived variables
TxPower_dBW = TxPower_dBmW-30;
[thetaRx_deg,phiRx_deg,~] = Coord2ThetaPhi(CoordTx,CoordRx);
[thetaTx_deg,phiTx_deg,d] = Coord2ThetaPhi(CoordRx,CoordTx);
%% Calcualte antenna Gain
% Transmitter Gain 
GainTx = AEmodel(thetaTx_deg, phiTx_deg,thetaTxBoreSight_deg, phiTxBoreSight_deg,...
    BeamWidthTx_deg,SideLobeStrengthTx,AEtypeTx);
GainTx_dB = pow2db(GainTx);
% Receiver Gain
GainRx = AEmodel(thetaRx_deg, phiRx_deg,thetaRxBoreSight_deg, phiRxBoreSight_deg,...
    BeamWidthRx_deg,SideLobeStrengthRx,AEtypeRx);
GainRx_dB = pow2db(GainRx);
% Total Gain
Gain_dB = GainTx_dB + GainRx_dB;
%% Channel Effects
h1 = CoordRx(3);h2 = CoordTx(3);
theta = thetaRx_deg; 
[~,~,AbsorptionLoss_dB,PLtotal_dB,~] = FullOutput(Freq_THz,h1,h2,d,theta);
end
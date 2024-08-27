%% Optimize Beamwidth for Max Capacity
function [BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
    SideLobeStrengthRx_max,SideLobeStrengthTx_max,ProbAli] = ...
    AntennaGainOptim(BeamWidthRx_deg,BeamWidthTx_deg,...
    SideLobeStrengthRx,SideLobeStrengthTx,TxPower_dBmW,...
            CoordRx,CoordTx, AEtypeRx,AEtypeTx,...
            thetaTxBoreSight_deg,phiTxBoreSight_deg,...
            thetaRxBoreSight_deg,phiRxBoreSight_deg,...
            maskIdx,Freq_THz)

A = 30;
mean_theta = thetaRxBoreSight_deg;      
mean_phi = phiRxBoreSight_deg;               
std_dev_theta = 10;     
std_dev_phi = 10;           
num_samples = 10;         
thetaRxBoreSight = mean_theta + std_dev_theta * randn(num_samples, 1);
phiRxBoreSight = mean_phi + std_dev_phi * randn(num_samples, 1);
sample_index = randi(num_samples); 
thetaRxBoreSight_deg = thetaRxBoreSight(sample_index);
phiRxBoreSight_deg = phiRxBoreSight(sample_index);

initialTemperature = 1e13;
finalTemperature = 1e10;
coolingRate = 0.95;
maxIterations = 10;

PhiTxInit = BeamWidthTx_deg(1);
ThetaTxInit = BeamWidthTx_deg(2);
bestPhiTx = PhiTxInit;
bestThetaTx = ThetaTxInit;
currentTemperature = initialTemperature;

[AbsorptionLoss_dB, PLtotal_dB, Gain_dB, TxPower_dBW] = ...
    Distance_to_Loss(TxPower_dBmW, CoordRx, CoordTx, AEtypeRx, BeamWidthRx_deg, SideLobeStrengthRx, AEtypeTx, BeamWidthTx_deg, SideLobeStrengthTx, thetaTxBoreSight_deg, phiTxBoreSight_deg, thetaRxBoreSight_deg, phiRxBoreSight_deg, Freq_THz);

[Cap, ~, ~] = Mask_Cap(maskIdx, Freq_THz, PLtotal_dB, AbsorptionLoss_dB, Gain_dB, TxPower_dBW);
GainTx = SideLobeStrengthTx;
GainTx_dB = pow2db(GainTx);
GainRx = SideLobeStrengthRx;
GainRx_dB = pow2db(GainRx);
Gain_dB_misaligned = GainTx_dB + GainRx_dB;
[Cap_misaligned, ~, ~] = Mask_Cap(maskIdx, Freq_THz, PLtotal_dB, AbsorptionLoss_dB, Gain_dB_misaligned, TxPower_dBW);
ProbPhiTx = AlignmentProbability(std_dev_phi,PhiTxInit,BeamWidthTx_deg(1),A);
ProbThetaTx = AlignmentProbability(std_dev_theta,ThetaTxInit,BeamWidthTx_deg(2),A);
ProbAli = ProbPhiTx * ProbThetaTx;
bestRate = ProbAli * Cap + (1-ProbAli) * Cap_misaligned;
numNeighbors = 3;
for iter = 1:maxIterations
    bestBeamWidthTx_deg = [bestPhiTx,bestThetaTx];
    neighbors = generateNeighbor(bestBeamWidthTx_deg, numNeighbors);
    bestNeighbor = findBestNeighbor(neighbors, ThetaTxInit, PhiTxInit,std_dev_theta,std_dev_phi,A);
    BeamWidthTx_deg = bestNeighbor;
    newPhiTx = BeamWidthTx_deg(1);
    newThetaTx = BeamWidthTx_deg(2);
    [AbsorptionLoss_dB, PLtotal_dB, Gain_dB, TxPower_dBW] = ...
    Distance_to_Loss(TxPower_dBmW, CoordRx, CoordTx, AEtypeRx, BeamWidthRx_deg, SideLobeStrengthRx, AEtypeTx, BeamWidthTx_deg, SideLobeStrengthTx, thetaTxBoreSight_deg, phiTxBoreSight_deg, thetaRxBoreSight_deg, phiRxBoreSight_deg, Freq_THz);

    [Cap, ~, ~] = Mask_Cap(maskIdx, Freq_THz, PLtotal_dB, AbsorptionLoss_dB, Gain_dB, TxPower_dBW);
    ProbPhiTx = AlignmentProbability(std_dev_phi,PhiTxInit,newPhiTx,A);
    ProbThetaTx = AlignmentProbability(std_dev_theta,ThetaTxInit,newThetaTx,A);
    ProbAli = ProbPhiTx * ProbThetaTx;
    newRate = ProbAli * Cap + (1-ProbAli) * Cap_misaligned;
    rateDifference = newRate - bestRate;

    if rateDifference > 0 || exp(rateDifference / currentTemperature) > rand()
        bestThetaTx = newThetaTx;
        bestPhiTx = newPhiTx;
        bestRate = newRate;
    end
    currentTemperature = currentTemperature * coolingRate;
    if currentTemperature < finalTemperature
        break;
    end
end

BeamWidthRx_deg_max = BeamWidthRx_deg;
BeamWidthTx_deg_max = [bestPhiTx,bestThetaTx];
SideLobeStrengthRx_max = SideLobeStrengthRx;
SideLobeStrengthTx_max = SideLobeStrengthTx;
end
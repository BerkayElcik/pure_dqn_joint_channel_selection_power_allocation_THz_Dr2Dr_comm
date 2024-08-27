%% WaterFilling Power allocation
function [PowerNBProdNsub_dBW,maskOut] = WaterFillingAlgo(maskIn,AbsorptionLoss_dB,PLtotal_dB,TotalTxPower_dBW,Freq_THz)

Idx = find(maskIn==1);
DeltaFreqHz = DeltaFreq(Freq_THz.*1e12);

ReqDeltaFreqHz = DeltaFreqHz(Idx);

ReqAbsorptionLoss_dB = AbsorptionLoss_dB(Idx);

PnoiseReqFreq = NoisePower(ReqAbsorptionLoss_dB,ReqDeltaFreqHz);

AtotalReqFreq = db2pow(PLtotal_dB(Idx));
P_Total = db2pow(TotalTxPower_dBW);
errK = 1;
AN_ReqFreq = PnoiseReqFreq.*AtotalReqFreq;
AN_ReqFreq_dummy = AN_ReqFreq;
AN_ReqFreq_dummy(AN_ReqFreq_dummy==0) = [];
minAN_ReqFreq_dummy = min(AN_ReqFreq_dummy(AN_ReqFreq_dummy ~= 0));
% Replace zeros in AN_ReqFreq with the minimum non-zero value scaled by 1e-2
AN_ReqFreq(AN_ReqFreq == 0) = minAN_ReqFreq_dummy * 1e-2;

% For initialization
Kold = db2pow(mean(pow2db(AN_ReqFreq)));% Linear Scale

fa = find(AN_ReqFreq<Kold);

if Kold == Inf
    disp('Inf geldi');
end
while errK > 1e-5
    n = length(fa);
    K = (P_Total/n) + (1/n*sum(AN_ReqFreq(fa),'all'));
    fa = find(AN_ReqFreq<K);
    errK = abs(Kold-K)/Kold;
    Kold = K;
    if Kold == Inf
        disp('Inf geldi');
    end
end
Sopt = zeros(size(AN_ReqFreq));
Sopt(fa) = K-AN_ReqFreq(fa);
SoptRegProdNsub = Sopt/sum(Sopt)*P_Total*length(fa);
SoptReg = Sopt/sum(Sopt)*P_Total;
maskOut = zeros(size(maskIn),'like',maskIn);
maskOut(Idx(fa)) = 1;

PowerNBProdNsub_dBW = zeros(size(maskIn));
PowerNBProdNsub_dBW(Idx) = (SoptRegProdNsub);
PowerNBProdNsub_dBW = pow2db(PowerNBProdNsub_dBW);

end
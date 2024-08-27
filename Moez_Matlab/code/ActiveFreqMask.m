 function maskIdx = ActiveFreqMask(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW,ReqIdx_Init)

[Cap_Init,ReqFreq_THz_Init,SNRout_Init] = Mask_Cap(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW);

ReqIdx_Init = find(maskIdx_Init==1);
SNRout_Sort = sort(SNRout_Init,'descend');
StartIdx = 1;
xData = [];yData = [];
TxPower_W = db2pow(TxPower_dBW)/length(ReqIdx_Init);
TotalTxPower_W = sum(TxPower_W);
N = 100;
% crude
for jj = StartIdx:N:length(SNRout_Sort)
    maskIdx = zeros(size(Freq_THz));
    SNR_jj = SNRout_Sort(jj);
    Idx_jj = find(SNRout_Init >=SNR_jj);
    ReqIdx_jj = ReqIdx_Init(Idx_jj);
    maskIdx(ReqIdx_jj) = 1;
    TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
    [Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
    xData = [xData,jj];
    yData = [yData,Cap_jj];

end

% to cater for the last value that might not be included in the crude
jj = length(SNRout_Sort);
maskIdx = zeros(size(Freq_THz));
SNR_jj = SNRout_Sort(jj);
Idx_jj = find(SNRout_Init >=SNR_jj);
ReqIdx_jj = ReqIdx_Init(Idx_jj);
maskIdx(ReqIdx_jj) = 1;
TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
[Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
xData = [xData,jj];
yData = [yData,Cap_jj];

[~,maxIdx_ydata] = max(yData);
maxIdx = xData(maxIdx_ydata);
StartIdx = maxIdx-N-StartIdx+1;
if StartIdx<1
    StartIdx = 1;
end

if maxIdx+N > length(SNRout_Sort)
    EndIdx = length(SNRout_Sort);
else
    EndIdx = maxIdx+N;
end


xData = [];yData = [];

for jj = StartIdx:1:EndIdx
    maskIdx = zeros(size(Freq_THz));
    SNR_jj = SNRout_Sort(jj);
    Idx_jj = find(SNRout_Init >=SNR_jj);
    ReqIdx_jj = ReqIdx_Init(Idx_jj);
    maskIdx(ReqIdx_jj) = 1;
    TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
    [Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
    xData = [xData,jj];
    yData = [yData,Cap_jj];
end

maskIdx = zeros(size(Freq_THz));
[~,maxIdx_yData] = max(yData);
Idx_Max = xData(maxIdx_yData);
SNR_max = SNRout_Sort(Idx_Max);
Idx_MaxT = find(SNRout_Init >=SNR_max);
ReqIdx_Max = ReqIdx_Init(Idx_MaxT);
maskIdx(ReqIdx_Max) = 1;

end
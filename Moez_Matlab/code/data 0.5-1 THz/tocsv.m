clc;clear;close all

freqs=load("ReqFreq_THz_0.5_1.mat");
%freqs=freqs.ReqFreq_THz;
freqs=freqs.Freq_THz;
loss_matrix=zeros(11,3000);
noise_matrix=zeros(11,3000);
for i=1:11
    data_path='variables_CFB_EP_0.5_1_distance=%d.mat';
    data_path = sprintf(data_path,i);
    FileData = load(data_path);
    loss=FileData.Atotal;
    noise_power=FileData.Pnoise;
    loss_matrix(i,:)=loss;
    noise_matrix(i,:)=noise_power;
end

writematrix(freqs, 'freqs_0.5_1.csv')
writematrix(loss_matrix, 'loss_matrix_0.5_1.csv')
writematrix(noise_matrix, 'noise_matrix_0.5_1.csv')
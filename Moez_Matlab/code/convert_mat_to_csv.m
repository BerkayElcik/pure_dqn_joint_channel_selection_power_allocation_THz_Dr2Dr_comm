clc;clear;close all

freqs=load("ReqFreq_THz_CFB_EP_0.75_0.8_res=1GHz.mat");
freqs=freqs.ReqFreq_THz;
loss_matrix=zeros(11,50);
noise_matrix=zeros(11,50);
for i=1:11
    data_path='variables_CFB_EP_0.75_0.8_res=1GHz_distance=%d.mat';
    data_path = sprintf(data_path,i);
    FileData = load(data_path);
    loss=FileData.Atotal;
    noise_power=FileData.Pnoise;
    loss_matrix(i,:)=loss;
    noise_matrix(i,:)=noise_power;
end

writematrix(freqs, 'freqs_0.75_0.8.csv')
writematrix(loss_matrix, 'loss_matrix_0.75_0.8.csv')
writematrix(noise_matrix, 'noise_matrix_0.75_0.8.csv')
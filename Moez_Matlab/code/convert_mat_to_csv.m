clc;clear;close all
FileData = load('ReqFreq_THz_CFB_EP_0.75_1.mat');
writematrix(FileData.ReqFreq_THz,'ReqFreq_THz_CFB_EP_0.75_1.csv') 

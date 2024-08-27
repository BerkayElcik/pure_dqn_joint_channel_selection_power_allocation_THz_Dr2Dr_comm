clc;clear;close all
FileData1 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=1.mat');
f1=FileData1.Atotal;

FileData2 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=2.mat');
f2=FileData2.Atotal;

FileData3 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=3.mat');
f3=FileData3.Atotal;

FileData4 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=4.mat');
f4=FileData4.Atotal;

FileData5 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=5.mat');
f5=FileData5.Atotal;

FileData6 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=6.mat');
f6=FileData6.Atotal;

FileData7 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=7.mat');
f7=FileData7.Atotal;

FileData8 = load('variables_CFB_EP_0.75_0.8_res=1GHz_distance=8.mat');
f8=FileData8.Atotal;

a=sum(f1-f2)
b=sum(f1-f3)
c=sum(f1-f4)
d=sum(f1-f5)
e=sum(f1-f6)
f=sum(f1-f7)
g=sum(f1-f8)

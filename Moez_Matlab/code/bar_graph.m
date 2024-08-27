clc;clear;close all
FileData = load('ReqFreq_THz_CFB_EP_0.75_0.8.mat');
data=FileData.ReqFreq_THz;
data=round(data,4);
data = unique(data(:).');
x=0.5:0.0001:1;

% Create a result vector with the same length as A, initialized to 0
y = zeros(1, length(x));

% Loop through the vector A
for i = 1:length(x)
    % Check if the current element in A is also in B
    if ismember(x(i), data)
        y(i) = 1;
    end
end
freqs=x(y==1);

% Display the result
disp('Result vector:');
disp(y);
disp(sum(y));
disp(freqs);
bar(freqs)
difs=diff(freqs);
bar(difs)
a=difs(difs>0.0001);
sum(difs>0.0001)
mask=difs>0.0001;
mask2=difs<=0.0001;

non_cfb_freqs=x(mask);
cfb_freq=x(mask2);
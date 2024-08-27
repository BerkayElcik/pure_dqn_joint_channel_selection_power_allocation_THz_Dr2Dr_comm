clc;clear;close all
FileData = load('ReqFreq_THz_CFB_EP_0.75_1.mat');
data=FileData.ReqFreq_THz;
x=0.75:0.0001:1;

% Create a result vector with the same length as A, initialized to 0
y = zeros(1, length(x));

% Loop through the vector A
for i = 1:length(x)
    % Check if the current element in A is also in B
    if ismember(x(i), data)
        result(i) = 1;
    end
end

% Display the result
disp('Result vector:');
disp(result);
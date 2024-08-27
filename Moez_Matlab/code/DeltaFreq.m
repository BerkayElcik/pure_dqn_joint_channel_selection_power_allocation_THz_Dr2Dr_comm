function deltaFreq = DeltaFreq(Freq)
    if numel(Freq) == 1
        deltaFreq = 3.120000008493652e+08;
    else
        deltaFreq = diff(Freq);
        deltaFreq = [deltaFreq(1); deltaFreq(:)]; % Ensure deltaFreq is a column vector
        deltaFreq = (deltaFreq(1:end-1) + deltaFreq(2:end)) / 2;
        deltaFreq = [deltaFreq; deltaFreq(end)]; % Ensure deltaFreq remains a column vector
    end
    deltaFreq = deltaFreq';
end

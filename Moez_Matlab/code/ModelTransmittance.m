function AbsLoss_dB = ModelTransmittance(f, h1, d)
    % Calculate Loss from Model
    % Input:
    % f: Frequency 0.79025-0.90974, 0.93002-0.93969 THz
    % h1: Height of the transmitter (TX)
    % d: Distance between TX and RX

    % Define frequency bands
    Band1 = [0.79025, 0.90974]; % Band-1: 0.79025-0.90974 THz
    Band2 = [0.93002, 0.93969]; % Band-2: 0.93002-0.93969 THz

    if ~all((f >= Band1(1) & f <= Band1(2)) | (f >= Band2(1) & f <= Band2(2)))
        error('The input frequency is out of modeled bands');
    end
    
    % Coefficients for Band-1
    CoeffExpFit_Band1_B1_A1 = [-78091746.6392514; 526711033.070705; -1553926236.73309; 2619168573.80666; -2758609352.71924; 1859136533.32764; -782938256.675700; 188374328.578433; -19824900.2919532];
    CoeffExpFit_Band1_B1_B1 = -0.000411249080709157;
    
    a = CoeffExpFit_Band1_B1_A1' * (f .^ [8; 7; 6; 5; 4; 3; 2; 1; 0]);

    b = CoeffExpFit_Band1_B1_B1;

    % Compute loss
    bH1 = a .* exp(b .* h1);
    loss = exp(bH1 .* d); 
    AbsLoss_dB = 10.* log10(loss);

end

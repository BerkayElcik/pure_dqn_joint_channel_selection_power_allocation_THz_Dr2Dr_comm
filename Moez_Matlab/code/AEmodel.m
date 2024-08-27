%% Antenna Model
function Gain = AEmodel(thetaIn, phiIn,thetaBoreSight, phiBoreSight,...
    BeamWidth,SideLobeStrength,AEtype)
% AEtype : 0 - Isotropic
%          1 - Omin-directional
%          2 - Sector AE 3-D assumed
%          3 - 
    if length(BeamWidth) == 1
        BeamWidth = [BeamWidth,BeamWidth];
    elseif length(BeamWidth) > 2 || length(BeamWidth) < 1
        error('Wrong parameter for length(BeamWidth)')
    end
if phiBoreSight == 0 || phiBoreSight == 180
    delta_theta = 0;
else
    delta_theta = AcuteAngle(thetaBoreSight,thetaIn);    
end
%     delta_theta = mod(abs(thetaBoreSight - thetaIn),180); %Original
delta_phi = AcuteAngle(phiBoreSight,phiIn);
%     delta_phi = abs(phiBoreSight - phiIn);
    Cond = @(t,p) double((BeamWidth(1)/2 >= t)&&(BeamWidth(2)/2 >=p));
%     Cond = @(t,p) double((BeamWidth(1)/2 >= t)&&(BeamWidth(1)/2 >=p));
    switch AEtype
        case 0 % Isotropic antenna
            Gain = 1;
        case 1 % Omni directional antenna
            Gain = 2;
        case 2 % WCNC conference Antenna Model
            BmW = sum(deg2rad(BeamWidth));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    ((2*pi - (2*pi - BmW)*SideLobeStrength) ./ BmW) .* ...
                    Cond(delta_theta,delta_phi);
        case 3 % Spherical Lobe Model WO SLL Subtraction
            BmW_Phi = (deg2rad(BeamWidth(1)));
            BmW_theta = (deg2rad(BeamWidth(2)));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    4*pi/(BmW_Phi*BmW_theta).* ...
                    Cond(delta_theta,delta_phi);
        case 4 % Spherical Lobe Model with side lobe gain subtraction
            BmW_Phi = (deg2rad(BeamWidth(1)));
            BmW_theta = (deg2rad(BeamWidth(2)));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    (4*pi-(4*pi-BmW_Phi-BmW_theta)*SideLobeStrength)/(BmW_Phi*BmW_theta).* ...
                    Cond(delta_theta,delta_phi);
        case 5 % Spherical Lobe Model with Alignment Probability
            BmW_Phi = (deg2rad(BeamWidth(1)));
            BmW_theta = (deg2rad(BeamWidth(2)));
            Gain = 4*pi/(BmW_theta*BmW_Phi);
        otherwise 
            Gain = 1;
    end
end
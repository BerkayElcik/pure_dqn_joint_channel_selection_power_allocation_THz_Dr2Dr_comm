function [Ltotal_dB] = AtmAbsorption(Freq_THz, h1, theta)

% Freq_THz is the frequency vector in THz
% h1 is the observation point altitude in m
% theta is the radio elevation angle (the geometric and radio angles differ due to refraction)

%% Calculating layers thickness according to ITU-R P.676-12
n_lower = floor(100*log(1e4*h1 * (exp(1/100)-1) + 1) + 1);
n = n_lower:922;
delta = 0.0001 * exp((n - 1) / 100); % in km
 
r = 6371 + cumsum(delta); % radii of the layers
h = h1 + cumsum(delta); % heights from the surface
 
%% Approximate refraction index in the layers from ITU R P.453
N0 = 315;
h0 = 7.35;
n = 1 + N0 * 1e-6 * exp(-h / h0); % from equation (8)
 
%% Zenith angles in the layers
beta = zeros(1, length(n));
beta(1) = 90 - theta;  % depression angle
for i = 1:length(n)-1
    beta(i+1) = asind(n(i) * r(i) / (n(i+1) * r(i+1)) * sind(beta(i)));
end

%% Path segment lengths
a = -r .* cosd(beta) + sqrt(r.^2 .* cosd(beta).^2 + 2 * r .* delta + delta.^2);
 
%% Approximate water vapour density according to equation (6) in  ITU-R P.835-2 model
% standard ground vapour density 7.5 g/m3, ho = 2 km
rho_o = 7.5; h_o = 2;
rho = rho_o * exp(-h / h_o);
 
%% Obtain the standard temperature and Pressure according to the International atmospheric model
[T, ~, P, ~] = atmosisa(h * 1000);

%% Calculating the layers absorption for each frequency point
L = zeros(length(Freq_THz), length(a));
for f = 1:length(Freq_THz)
    for i = 1:length(a)
        L(f, i) = gaspl(a(i) * 1000, Freq_THz(f) * 1e12, T(i) - 273.15, P(i), rho(i));
    end
end

%% Total absorption for each frequency point
Ltotal = sum(L, 2);
%% Convert absorption to decibels (dB)
Ltotal_dB = 10 .* log10(Ltotal);
Ltotal_dB = Ltotal_dB';
end

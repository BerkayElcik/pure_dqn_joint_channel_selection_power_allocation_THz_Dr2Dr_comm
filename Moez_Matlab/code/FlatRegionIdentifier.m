%% Flat Band Identifier
function [No] = FlatRegionIdentifier(N,FlatnessCriteria,minNumberOfPoints)
% Input:
% N: Input Signal
% FlatnessCriteria : Vertical threshold
% minNumberOfPoints: minimum number of consecutive points to be considered
% DisplayFlag      : Plot command for plotting
% Output:
% No               : Indices of flat regions
%
%
% Example
% For path loss
% FlatnessCriteria = 0.5;% dBm scale
% NumberOfPoints = 10;
% [Idx_Total_PL_Flat] = FlatRegionIdentifier(Atotal_dB(Atotal_dB<=200),FlatnessCriteria,NumberOfPoints,DisplayFlag);
% Example
% For noise
% [Idx_NoiseFlat] = FlatRegionIdentifier(pow2db(Pnoise*1e3),FlatnessCriteria,NumberOfPoints,DisplayFlag);

%% Identify indices corresponding to Minimum and maximum number of points criteria
FullBandFlag = false; % Assign 0s
Threshold = FlatnessCriteria; % dB scale total variation.
Na = zeros(length(N),'uint16'); % Initial Na with 0s up to the length of N
for ii = 1:length(N) % for loop for the entire length of N
    Na(ii,ii) = true; % generate a matrix (length N x length N) and assign 1s diagonal of the matrix
    Nb1 = (abs(N(ii)-N)<=Threshold/2); % all entries from ii before and after ii of the matrix which are within the threshold
    IdxNb1 = find(Nb1==false); % find if there are entries which are 0s (i.e. out of the threshold)
    if ~isempty(IdxNb1) % if there are 0s entries
        IdxNb1_ii_less = IdxNb1(find(IdxNb1<ii));% find all entries less than considered point that are out of threshold
        IdxNb1_ii_greater = IdxNb1(find(IdxNb1>ii));% find all entries greater than considered point that are out of threshold
        IdxNb1_ii_less = sort(IdxNb1_ii_less,'descend');
        IdxNb1_ii_greater = sort(IdxNb1_ii_greater,'ascend');
        
        if ii>1 && ~isempty(IdxNb1_ii_less)
            Nb1(1:IdxNb1_ii_less(1)) = false; % Less index entries that are not connecting to consdered point and are in threshold are set to zero
        end
        if ii<length(N) && ~isempty(IdxNb1_ii_greater)
            Nb1(IdxNb1_ii_greater(1):end) = false;% Greater index entries that are not connecting to consdered point and are in threshold are set to zero
        end
    else
        if ii == 1 % first entry (iteration)
            FullBandFlag = true; % FullBandFlag = 1
            break;
        else
            FullBandFlag = false; % FullBandFlag = 1
        end
    end
    Na(ii,:) = Nb1;
end
if ~FullBandFlag % If FullBandFlag is False (i.e. 0)
    N2 = sum(Na,2);
    No = find(N2>=minNumberOfPoints);
else
    No = 1:length(N); % If FullBandFlag is True (i.e. 1), which means the entire band is flat, return the index of the center point of the band
end
end
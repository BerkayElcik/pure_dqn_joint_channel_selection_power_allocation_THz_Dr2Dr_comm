function [Theta21_deg,Phi21_deg,d] = Coord2ThetaPhi(Coord1,Coord2)
dVect_12 = Coord1-Coord2;
d = sqrt(sum(dVect_12.^2));
Phi21_deg = acosd(dVect_12(3)./d);
Theta21_deg = mod(360+atan2d(dVect_12(2),dVect_12(1)),360);
end
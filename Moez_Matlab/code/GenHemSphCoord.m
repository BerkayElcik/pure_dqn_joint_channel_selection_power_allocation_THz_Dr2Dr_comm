%% Generate Hemisphere coordinates
function XYZ = GenHemSphCoord(XYZ_Tx,XYZ_Rx)
r = 20;%meters
theta_d = 90;%[0:10:180];
phi_d = [0:10:180];%[90];
%XYZ_Tx = [50,50,100];

%[theta_d,phi_d,r] = Coord2ThetaPhi(XYZ_Tx,XYZ_Rx);

XYZ = [];
for pp = 1:size(XYZ_Tx,1)
    for ii = 1:length(r)
        for jj = 1:length(phi_d)
            for kk = 1:length(theta_d)
                x = r(ii).*cosd(theta_d(kk))*sind(phi_d(jj))+XYZ_Tx(pp,1);
                y = r(ii).*sind(theta_d(kk))*sind(phi_d(jj))+XYZ_Tx(pp,2);
                z = r(ii).*cosd(phi_d(jj))+XYZ_Tx(pp,3);
                XYZ = [XYZ;[XYZ_Tx(pp,1),XYZ_Tx(pp,2),XYZ_Tx(pp,3),x,y,z,theta_d(kk),phi_d(jj)]];
            end
        end
    end
end
end
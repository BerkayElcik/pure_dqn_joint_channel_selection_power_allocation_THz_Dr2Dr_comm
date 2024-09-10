clear; clc; close all
% Transmitted Power
TxPower_dBmW = 24;% dBm;
%% TVT 3D antenna continuous Frequency Range
StartFreqTHz = 0.5;
StopFreqTHz = 1;
NumPoints = 3000;
Freq_THz = linspace(StartFreqTHz, StopFreqTHz, NumPoints); % 
%% Trajectory parameters
RealSimulatedFlag = 1;% 0: Real, 1: Simulated
LinearSphereFlag = 0;% 0: Linear, 1: Sphere
ThetaPhiFlag = 0;% 0: Theta Direction, 1: Phi Direction
StraightDiagonalFlag = 0;% 0: Straight, 1: Diagonal
%% Setting inital corrdinates for drones
x_t = 0; y_t = 10; z_t = 100;
x_r = 0; y_r = 11; z_r = 100; % for theta/phi, d = 20 m
%% Antenna parameters
% Receiver
AEtypeRx = 3;
BeamWidthRx_deg = [10,10]; 
SideLobeStrengthRx = 0.1;
% Transmitter
AEtypeTx = AEtypeRx;

BeamWidthTx_deg = [10,10];%0.181; %% mean linear gain = 647.6130, 7.9812 degrees
SideLobeStrengthTx = 0.1;

%% Antenna positions on drone
% Transmitter
thetaTxBoreSight_deg = -90;
phiTxBoreSight_deg = 90;
% Receiver
thetaRxBoreSight_deg = 90;
phiRxBoreSight_deg = 90;
%% Data of accelerations Tx Rx
if RealSimulatedFlag == 0
    load data_small_windy_average.mat
% vi = 0; % for small scale mobility case, i.e., just hovering
dt = mean(diff(DataNewAv(:,1)));% %5 msec sampling period
timeFinal = 5.5;
FinalIdx = round(timeFinal/dt);
LastIdx = floor(length(DataNewAv(:,1))/FinalIdx)*FinalIdx;
Data = DataNewAv(1:LastIdx,:);
DataR = reshape(Data,[FinalIdx,LastIdx/FinalIdx,size(Data,2)]);
Trace1Number = 2;
Trace2Number = 1;
DataTx = squeeze(DataR(:,Trace1Number,:));
DataRx = squeeze(DataR(:,Trace2Number,:));

acc_x_t = DataTx(:,2);
acc_y_t = DataTx(:,3);
acc_z_t = DataTx(:,4);
time_mob_t = DataTx(:,1)-DataTx(1,1);

idx_r = FinalIdx+1:FinalIdx*2;

acc_x_r = DataRx(:,2);
acc_y_r = DataRx(:,3);
acc_z_r = DataRx(:,4);
time_mob_r = DataRx(:,1)-DataRx(1,1);

elseif RealSimulatedFlag == 1
    DataTx = [(0:10)',zeros(11,3)];
    DataRx = [(0:10)',zeros(11,3)];
else
    error('RealSimulatedFlag == 0,1')
end

%% Using linear accelereations
if LinearSphereFlag == 0
    time_mob_t  = DataTx(:,1);
    acc_x_t     = DataTx(:,2);
    acc_y_t     = DataTx(:,3);
    acc_z_t     = DataTx(:,4);
    vi_x_t      = 0;
    vi_y_t      = 0;
    vi_z_t      = 0;

    time_mob_r  = DataRx(:,1);
    acc_x_r     = DataRx(:,2);
    acc_y_r     = DataRx(:,3);
    acc_z_r     = DataRx(:,4);

    if StraightDiagonalFlag == 0
        vi_x_r      = 0;
        vi_y_r      = 10;
    elseif StraightDiagonalFlag == 1
        vi_x_r      = sqrt(2)*5;
        vi_y_r      = sqrt(2)*5;
    else
        error('StraightDiagonalFlag == 0,1')
    end
vi_z_r      = 0;
%% Using Accelerations for constant trajectory
[~,x_mobility_t] = a2vd(acc_x_t,time_mob_t,vi_x_t,x_t);
[~,y_mobility_t] = a2vd(acc_y_t,time_mob_t,vi_y_t,y_t);
[~,z_mobility_t] = a2vd(acc_z_t,time_mob_t,vi_z_t,z_t);

[~,x_mobility_r] = a2vd(acc_x_r,time_mob_r,vi_x_r,x_r);
[~,y_mobility_r] = a2vd(acc_y_r,time_mob_r,vi_y_r,y_r);
[~,z_mobility_r] = a2vd(acc_z_r,time_mob_r,vi_z_r,z_r);

%% Using sphere motion
elseif LinearSphereFlag == 1 
    if RealSimulatedFlag == 1
        % x_t = 0; y_t = 10; z_t = 100;
        % x_r = 0; y_r = 11; z_r = 100;
        if ThetaPhiFlag == 0
            theta_d = 0:10:180;%[0:10:180];
            phi_d = [90];
        elseif ThetaPhiFlag == 1
            theta_d = 180;%[0:10:180];
            phi_d   = 0:10:180;%[90];
        else
            error('ThetaPhiFlag == 0,1');
        end
        r = sqrt(sum((x_t-x_r)^2+(y_t-y_r)^2+(z_t-z_r)^2));%meters
        XYZ_Tx = [x_t,y_t,z_t];
        XYZ_Rx = [x_r,y_r,z_r];
        
        XYZ = GenHemSphCoord(XYZ_Tx,XYZ_Rx);
        x_mobility_t = XYZ(:,1);
        y_mobility_t =  XYZ(:,2);
        z_mobility_t =  XYZ(:,3);
        x_mobility_r =  XYZ(:,4);
        y_mobility_r =  XYZ(:,5);
        z_mobility_r =  XYZ(:,6);
        time_mob_t = (0:(length(z_mobility_r)-1))*1;
    else
        error('RealSimulatedFlag == 1')
    end
else
    error('LinearSphereFlag == 0,1')
end
%% Plotting of figures
% figure(1);
% plot3(x_mobility_t,y_mobility_t,z_mobility_t,'*','MarkerSize',10)
% hold on
% plot3(x_mobility_r,y_mobility_r,z_mobility_r,'.','MarkerSize',20)
% xlabel('x coordinate [m]')
% ylabel('y coordinate [m]')
% zlabel('z coordinate [m]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% legend('Tx Drone Trail','Rx Drone Trail','FontSize',12)
% grid on;axis square;
% drawnow
% hold off
%% 
 d_t_r_mob = sqrt((x_mobility_r-x_mobility_t).^2+...
     (y_mobility_r-y_mobility_t).^2+...
    (z_mobility_r-z_mobility_t).^2);
% figure(2);
% plot(time_mob_t,d_t_r_mob,'-*','LineWidth',5,'MarkerSize',10)
% xlabel('Time (t) [sec]')
% ylabel({'Instantaneous distance (d_{mob})';'between Tx & Rx drones [m]'})
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% grid on;axis square;
% legend('Tx-Rx','FontSize',12)
% drawnow
% hold off
% %
DataOutT = zeros(length(x_mobility_t),1);

% Calcualte intial antenna alignment
cTx = [x_mobility_t,y_mobility_t,z_mobility_t];
cRx = [x_mobility_r,y_mobility_r,z_mobility_r];
[thetaTxBoreSight_deg,phiTxBoreSight_deg, ...
    thetaRxBoreSight_deg,phiRxBoreSight_deg] = BorsightCalc(cTx(1,:),cRx(1,:));
% Drone coordinates
FreqTHz_TxT = cell(length(x_mobility_t),1);
figure(3);
xlabel('Time (t) [sec]')
ylabel('Capacity (C) [Gbps]')
hold on;
for kk = 1:length(x_mobility_t)
    CoordTx = [x_mobility_t(kk),y_mobility_t(kk),z_mobility_t(kk)];
    CoordRx = [x_mobility_r(kk),y_mobility_r(kk),z_mobility_r(kk)];
    BandWidthOptFlag = 0;% 0: STD, 1: OA, 2: CFB
    BeamWidthOptFlag = 0;% 0: WBO, 1: BO
    PowerAllocFlag = 0;% 0: EP, 1: WF
    [CapOut,FreqTHz_Tx,BeamWidthRx_deg_max,...
    BeamWidthTx_deg_max,SNRout_dB,GainTotal,Rate] = Distance_to_Cap_Opt(...
        PowerAllocFlag,TxPower_dBmW,...
        CoordRx,CoordTx,...
        StartFreqTHz,StopFreqTHz,...
        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
        BandWidthOptFlag,BeamWidthOptFlag,Freq_THz);
    %% 
    [thetaRx_deg,phiRx_deg] = Coord2ThetaPhi(CoordTx,CoordRx);
    delta_thetaRx_deg = AcuteAngle(thetaRxBoreSight_deg,thetaRx_deg);
    delta_phiRx_deg = AcuteAngle(phiRxBoreSight_deg,phiRx_deg);
    [thetaTx_deg,phiTx_deg] = Coord2ThetaPhi(CoordRx,CoordTx);
    delta_thetaTx_deg = AcuteAngle(thetaTxBoreSight_deg,thetaTx_deg);
    delta_phiTx_deg = AcuteAngle(phiTxBoreSight_deg,phiTx_deg);
    FreqTHz_TxT{kk,1} = FreqTHz_Tx;
    DataOutT(kk,1:11) = [CapOut,BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
        GainTotal,delta_thetaRx_deg,delta_phiRx_deg,delta_thetaTx_deg,delta_phiTx_deg,Rate];
    plot(time_mob_t(1:kk),DataOutT(1:kk,1)/1e9);
%drawnow;
end

% % Plot Freq vs time
% figure(4)
% hold off
% MrkrSize= 10;
% for ll = 1:length(FreqTHz_TxT)
%     freqTHz_ll = FreqTHz_TxT{ll};
%     time_ll = ones(size(freqTHz_ll))*time_mob_t(ll);
%     plot(time_ll,freqTHz_ll,'.','MarkerSize',MrkrSize+2)
% hold on
% end
% xlabel('Time (t) [sec]')
% ylabel('Freq (THz)')
% title('Active Frequency Bands wrt Time')
% set(gca,'FontSize',14)
% grid on;axis square;
% hold off
% Plot Average Capacity
% Plot Average Capacity
figure(5);
semilogy(time_mob_t,DataOutT(:,1)/1e9,'g','LineWidth',5)
xlabel('Time (t) [sec]')
ylabel('Capacity (C) [Gbps]')
title('Tx Fixed - Rx Moving away with Constant Velocity')
set(gca,'FontSize',14)
legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
grid on;axis square;
drawnow
% 
% %% Beam Width VS Time
% figure(6);
% yyaxis right
% hold off
% yyaxis left
% hold off
% plot(time_mob_t,DataOutT(:,2),'.-y','LineWidth',9)
% hold on
% plot(time_mob_t,DataOutT(:,2),':k','LineWidth',6)
% hold on
% plot(time_mob_t,DataOutT(:,3),'-b','LineWidth',3)
% hold on
% plot(time_mob_t,DataOutT(:,3),'--r','LineWidth',3)
% hold on
% plot(time_mob_t,DataOutT(:,6),'--co','LineWidth',3,'MarkerSize',6)
% hold on
% plot(time_mob_t,DataOutT(:,5),'--gd','LineWidth',3,'MarkerSize',6)
% hold on
% ylabel('Angles [degrees]')
% yyaxis right
% plot(time_mob_t,DataOutT(:,4),'--r*','LineWidth',3,'MarkerSize',12)
% set(gca,'FontSize',14)
% xlabel('Time (t) [sec]')
% ylabel('Total Antenna Gain (G_{Total}) [dB]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% legend({'\psi_{\phi - Tx}','\psi_{\theta - Tx}',...
%     '\psi_{\phi - Rx}','\psi_{\theta - Rx}',...
%     '\Delta{\phi}_{\{Rx,Tx\}}','\Delta{\theta}_{\{Rx,Tx\}}',...
%     'G_{Total \{Rx,Tx\}}'},'FontSize',10)
% grid on;axis square;
% hold off;
% drawnow
% 
% % % Plot Freq vs time
% figure(7)
% MrkrSize= 10;
% for ll = 1:length(FreqTHz_TxT)
%     freqTHz_ll = FreqTHz_TxT{ll};
%     d_t_r_mob_ll = ones(size(freqTHz_ll))*d_t_r_mob(ll);
%     plot(d_t_r_mob_ll,freqTHz_ll,'.','MarkerSize',MrkrSize+2)
% hold on
% end
% xlabel('Distance [meters]')
% ylabel('Freq (THz)')
% title('Active Frequency Bands wrt Distance')
% set(gca,'FontSize',14)
% grid on;axis square;
% hold off
% 
% %
% % % Plot Capacity vs Distance
% figure(8);
% semilogy(d_t_r_mob,DataOutT(:,1)/1e9,'-r+','LineWidth',1)
% xlabel('Distance [meters]')
% ylabel('Capacity (C) [Gbps]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow

% Beam Width VS Distance
figure(9);
plot(time_mob_t, DataOutT(:, [2, 3]), 'LineWidth', 7); % Plot Beamwidth Rx
hold on
plot(time_mob_t, DataOutT(:, [4, 5]), 'LineWidth', 3); % Plot Beamwidth Tx
set(gca, 'FontSize', 14);
xlabel('time (t) [sec]');
ylabel('Beamwidth (deg)');
title('Misalignment via Diagonal Mobility');
legend({'\theta_{rx}', '\phi_{rx}', '\theta_{tx}', '\phi_{tx}'}, 'FontSize', 12);
grid on;
axis square;
drawnow;
hold off;
% Spectral Efficiency
% NB_count = cellfun(@length,FreqTHz_TxT);
% Total_Band = (NB_count.*3.120000008493652e+08);
% SE = DataOutT(:,1)./Total_Band;
% %
% %% Plot SE vs Distance
% figure(10);
% semilogy(d_t_r_mob,SE,'-y','LineWidth',1)
% xlabel('Distance [meters]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% %
% %% Plot SE vs Time
% figure(11);
% plot(time_mob_t,SE,'-y','LineWidth',1)
% xlabel('Time [sec]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% %% Beam Width VS Distance
% figure(12);
% plot(time_mob_t,DataOutT(:,2:4),'LineWidth',7)
% hold on
% plot(time_mob_t,DataOutT(:,2:4),'LineWidth',3)
% set(gca,'FontSize',14)
% xlabel('Time [sec]')
% ylabel('Beamwidth (deg)')
% title('Tx Fixed - Rx Moving')
% legend({'Beamwidth - Rx','Beamwidth - Tx'},'FontSize',12)
% grid on;axis square;
% drawnow
% hold off;
% 
%Capacity vs. Theta
% figure(13);
% semilogy(XYZ(:,7),DataOutT(:,1)/1e9,'g','LineWidth',5)
% xlabel('Azimuth Angle (\theta) [degrees]')
% ylabel('Capacity (C) [Gbps]')
% title('Tx Fixed - Rx Moving away')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% drawnow

% %Capacity vs. Elevation
% figure(13);
% semilogy(XYZ(:,8),DataOutT(:,1)/1e9,'g','LineWidth',5)
% xlabel('Elevation Angle (\phi) [degrees]')
% ylabel('Capacity (C) [Gbps]')
% title('Tx Fixed - Rx Moving away')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% drawnow
% % 
% figure(14);
% semilogy(XYZ(:,8),SE,'-y','LineWidth',1)
% xlabel('Elevation Angle (\phi) [degrees]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% 
% % Energy Efficiency
% figure(15);
% semilogy(time_mob_t,((DataOutT(:,1))/1e9)/0.25,'-r','LineWidth',1)
% xlabel('Time (t) [sec]')
% ylabel('Energy Efficiency (EE) [Giga bits/joules]')
% title('Tx - Rx Moving for With Practical Traces')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% 
% figure(16);
% semilogy(XYZ(:,8),((DataOutT(:,1))/1e9)/0.25,'-r','LineWidth',1)
% xlabel('Zenith Angle (\phi) [degrees]')
% xlabel('Elevation Angle (\phi) [degrees]')
% ylabel('Energy Efficiency (EE) [Giga bits/joules]')
% title('Tx - Rx Moving for With Practical Traces')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% 



function [v,d] = a2vd(a,t,v0,d0)
% Find velocity and distance at a given linear accelerations for each time
% stamp
N = length(a);
if length(t)~=N
    error('acc  ~=  time')
end
v = zeros(size(a));
d = zeros(size(a));
v(1) = v0;
d(1) = d0;
for kk = 2:N
    a_kk = (a(2:kk)+a(1:kk-1))/2;
    dt_kk = (t(2:kk)-t(1:kk-1));
    v(kk) = sum(a_kk.*dt_kk)+v0;
    d(kk) = d(kk-1) + v(kk-1)*dt_kk(end) + 0.5*a_kk(end)*dt_kk(end).^2;
end
end
%% Actute Angle calculator
function AngleOut = AcuteAngle(AngleIn1,AngleIn2)
diffAngle = mod(abs(AngleIn1-AngleIn2),360);
if diffAngle > 180 && diffAngle <= 360 
    AngleOut = 360 - diffAngle;
elseif diffAngle <= 180 && diffAngle >= 0 
    AngleOut = diffAngle;
else
    error('Wrong angles, there is some problem somwhere ...')
end
end
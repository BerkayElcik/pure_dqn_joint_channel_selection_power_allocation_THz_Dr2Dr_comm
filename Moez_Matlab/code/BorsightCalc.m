function [tTx,pTx, tRx,pRx] = BorsightCalc(cTx,cRx)
[tRx,pRx,~] = Coord2ThetaPhi(cTx,cRx);
[tTx,pTx,~] = Coord2ThetaPhi(cRx,cTx);
end

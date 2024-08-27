function bestNeighbor = findBestNeighbor(neighbors, ThetaTxInit, PhiTxInit, std_dev_theta, std_dev_phi, A)
    bestAlignmentProb = 0;
    bestNeighbor = [];

    for i = 1:size(neighbors, 1)
        neighbor = neighbors(i, :);
        Prob_phi = AlignmentProbability(std_dev_phi, PhiTxInit, neighbor(1), A);
        Prob_theta = AlignmentProbability(std_dev_theta, ThetaTxInit, neighbor(2), A);
        alignmentProb = Prob_phi * Prob_theta;
        
        if alignmentProb > bestAlignmentProb
            bestAlignmentProb = alignmentProb;
            bestNeighbor = neighbor;
        end
    end
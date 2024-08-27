function newValues = generateNeighbor(currentValues, numNeighbors)
    newValues = zeros(numNeighbors, numel(currentValues));

    perturbations = randn(numNeighbors, numel(currentValues)) * 3;
    for i = 1:numNeighbors
        newValues(i, :) = currentValues + perturbations(i, :);
        newValues(i, newValues(i, :) < 0) = 0;
        newValues(i, newValues(i, :) > 360) = 360;
    end
end

function Prob = AlignmentProbability(std_dev,alpha,alpha_prev,A)
     Gamma = sqrt(max((alpha_prev^2) / (A  * std_dev^2) - 0.5,0.01));
     term1 = erf(Gamma + alpha / (2 * sqrt(2) * std_dev));
     term2 = erf(Gamma - alpha / (2 * sqrt(2) * std_dev));
     Prob = 0.5 * (term1 - term2);
end

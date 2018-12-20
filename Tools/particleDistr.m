function  particleDistr = particleDistr(mPost, SPost, nr, np, ns, xMin, xMax)
%nr: number of rounds
%np: number of particles
%ns: number of trial points

bins = zeros(ns,nr+1); % We set up storage space for the number of bins.
bins(:,1) = floor(np/ns); % We divide the particles over the bins.
bins(1:mod(np,ns),1) = ceil(np/ns); % We give the first few bins an extra particle if it doesn't quite match.
dx = (xMax - xMin)/(ns - 1); % This is the distance between two trial points.

% We iterate over the number of rounds.
for i = 1:nr
    % We walk through all the bins.
    for j = 1:ns
        % For each particle, we make a comparison.
        for k = 1:bins(j,i)
            randomBin = ceil(rand(1,1)*ns); % We pick a new random bin.
            mut = mPost(j) - mPost(randomBin); % We set up the posterior distribution of f_j - f_r, with f_j being the current bin and f_r being the new random bin.
            St = SPost(j,j) + SPost(randomBin,randomBin) - 2*SPost(j,randomBin);
            sample = mut + sqrt(St)*randn(1,1); % This is a sample from f_j - f_r. If it is positive, then f_j > f_r and the particle can stay. If it is negative, then f_j < f_r and it should move to the new random bin.
            if sample >= 0
                bins(j,i+1) = bins(j,i+1) + 1;
            else
                bins(randomBin,i+1) = bins(randomBin,i+1) + 1;
            end
        end
    end
end

particleDistr = bins(:,i)/np/dx;

end
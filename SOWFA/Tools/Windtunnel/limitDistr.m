function limitDistr = limitDistr(mPost, SPost, ns, xMin, xMax)
%ns: number of sample points
nsReshape = size(mPost); 
% Next, we calculate the limit distribution of the particles.
P = zeros(ns,ns);
for i = 1:ns
    for j = 1:ns
        mut = mPost(i) - mPost(j);
        Sigmat = SPost(i,i) + SPost(j,j) - SPost(i,j) - SPost(j,i);
        P(i,j) = erf(mut/sqrt(2*Sigmat))/2 + 1/2;
    end
    P(i,i) = 1/2;
end

% We calculate the comparison matrix and use it to find the limit distribution of the particles.
mat = diag(diag(ones(ns,ns)*P)) - P;
outcome = zeros(ns,1);
mat(end,:) = ones(1,ns); % We set the bottom row equal to ones.
outcome(end) = 1; % We set the bottom element of the outcome equal to one.
limitDistr = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
limitDistr = limitDistr/prod((xMax - xMin)/(nsReshape(1) - 1)); % We turn the result into a PDF.
limitDistr = reshape(limitDistr, nsReshape(1,1), nsReshape(1,2)); % We put the result in a square format again.
end



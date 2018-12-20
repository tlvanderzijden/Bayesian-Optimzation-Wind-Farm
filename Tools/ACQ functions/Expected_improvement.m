%this file chooses an acquisition function based on the input parameter
function AQC=PI0(mPostUnc,SpostUnc,kappa);

fsh = max(mPostUnc); % This is \hat{f}^*.
xi = 0.2; % This is the exploration parameter for the PI and EI acquisition functions.
z0 = (mPostUnc - fsh)./sPostUnc;
zd = (mPostUnc - fsh - xi)./sPostUnc;


PI0 = normcdf(z0);
PId = normcdf(zd);
EI0 = (mPostUnc - fsh).*normcdf(z0) + sPostUnc.*normpdf(z0);
EId = (mPostUnc - fsh - xi).*normcdf(zd) + sPostUnc.*normpdf(zd);
EV = mPostUnc;
UCB1 = mPostUnc + 1*sPostUnc;
UCB2 = mPostUnc + 2*sPostUnc;
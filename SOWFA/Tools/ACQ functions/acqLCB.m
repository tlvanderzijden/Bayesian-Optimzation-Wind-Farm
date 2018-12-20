%this file creates the Lower confidence bound acquisition function, it
%recquires an extra kappa input that scales the variance
function val=acqLCB(mPostUnc,sPostUnc,kappa)

UCB = mPostUnc-kappa*sPostUnc;
val=UCB;
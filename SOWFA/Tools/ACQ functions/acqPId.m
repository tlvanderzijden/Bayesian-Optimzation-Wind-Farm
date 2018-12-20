%this file creates the Probability of Improvement acquisition function with
%exploration, it recquires an additional exploration parameter, Xi
function val=acqPId(mPostUnc,sPostUnc,xi)
fsh = max(mPostUnc); % This is \hat{f}^*.
zd = (mPostUnc - fsh - xi)./sPostUnc;
PId = normcdf(zd);
val=PId;
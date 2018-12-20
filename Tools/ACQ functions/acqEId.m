%this file creates the Expected Improvement acquisition function with
%exploration, it recquires an additional exploration parameter, Xi
function val=acqEId(mPostUnc,sPostUnc,xi)
fsh = max(mPostUnc); % This is \hat{f}^*.
zd = (mPostUnc - fsh - xi)./sPostUnc;
EId = (mPostUnc - fsh - xi).*normcdf(zd) + sPostUnc.*normpdf(zd);
val=EId;
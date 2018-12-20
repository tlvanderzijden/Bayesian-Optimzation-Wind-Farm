function SPost = postCovMatrix(hypcov, covfunc, xm, xs,sfh)
    %xm: measurement points: Dxn matrix 
    %xs: input points: Dxn matrix
    lx = hypcov(1); % This is the input length scale.
    lf = hypcov(2);  % This is the output length scale.
    
    X = [xm,xs]';
    ns = size(xs,2);
    nm = size(xm,2);
    K = feval(covfunc{:}, hypcov, X); %Compute covariance matrix
    KDivided = mat2cell(K,[nm,ns],[nm,ns]);
    Kmm = KDivided{1,1};
    Kms = KDivided{1,2};
    Ksm = KDivided{2,1};
    Kss = KDivided{2,2};
    mm = zeros(nm,1); % This is the prior mean vector of the measurement points.
    ms = zeros(ns,1); % This is the prior mean vector of the trial points.
    Sfh = sfh^2*eye(nm); % This is the noise covariance matrix.
    SPost = Kss - Ksm/(Kmm + Sfh)*Kms; % This is the posterior covariance matrix.
end
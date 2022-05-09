function theta = InitializeParam(X_train, Y_train, m)
    [d, ~] = size(X_train);
    [K, ~] = size(Y_train);
    
    W1 = randn(m, d)/sqrt(d);
    W2 = randn(K, m)/sqrt(m);
    
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    
    theta = {W1, W2, b1, b2};
end
function [H, P] = EvaluateClassifier(X, theta)
    W = theta(1:2);
    b = theta(3:4);
    
    s1 = W{1} * X + repmat(b{1}, [1, size(X, 2)]);
    H = max(0, s1);
    s = W{2} * H + repmat(b{2}, [1, size(X, 2)]);
    P = softmax(s);
end
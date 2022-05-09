function [J, loss] = ComputeCost(X, Y, W, b, lambda)
    theta = {W{1}, W{2}, b{1}, b{2}};
    [~, P] = EvaluateClassifier(X, theta);
    n = size(Y, 2);
    lcross = zeros(1, n);
    for i = 1:n
        lcross(i) = Y(:, i)' * log(P(:, i));
    end
    loss = - sum(lcross)/n;
    J = loss + lambda * (sum(W{1} .* W{1}, 'all') + sum(W{2} .* W{2}, 'all'));
end
function [J, loss] = ComputeCost(X, Y, NetParams, lambda)
    [~, P] = EvaluateClassifier(X, NetParams);
    n = size(Y, 2);
    lcross = zeros(1, n);
    W = NetParams.W;

    for i = 1:n
        lcross(i) = Y(:, i)' * log(P(:, i));
    end
    loss = - sum(lcross)/n;

    sumW = 0;
    for i=1:NetParams.k
        sumW = sumW + sum(W{i} .* W{i}, 'all');
    end

    J = loss + lambda * sumW;
end
function [J, loss] = ComputeCost(X, Y, NetParams, lambda, varargin)
    [~, P] = EvaluateClassifier(X, NetParams, varargin{1}, varargin{2});
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
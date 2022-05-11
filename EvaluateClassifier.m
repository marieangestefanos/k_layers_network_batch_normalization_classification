function [Xs, P] = EvaluateClassifier(X, NetParams)
    W = NetParams.W;
    b = NetParams.b;
    
    k = size(W, 2);

    s = W{1} * X + repmat(b{1}, [1, size(X, 2)]);
    Xs{1} = max(0, s);

    for i=2:(k-1)
        s = W{i} * Xs{i-1} + repmat(b{i}, [1, size(Xs{i-1}, 2)]);
        Xs{end + 1} = max(0, s);
    end

    % k - 1 = Nb of calculated Xs, input not included
    if size(Xs, 2) ~= (k-1)
        error("Error in EvaluateClassifier: size(Xs, 2) must equal k-1")
    end
    
    % Final linear transformation
    s = W{k} * Xs{k-1} + repmat(b{k}, [1, size(X, 2)]);

    P = softmax(s);
end
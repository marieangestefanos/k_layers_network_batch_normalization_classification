function accuracy = ComputeAccuracy(X, y, NetParams)
    n = size(y, 1);

    acc = zeros(1, n);

    [~, P] = EvaluateClassifier(X, NetParams);
    prediction = Argmax(P);

    for i = 1:n
        if prediction(i) == y(i)
            acc(i) = 1;
        else
            acc(i) = 0;
        end
    end

    accuracy = sum(acc)/n;
end
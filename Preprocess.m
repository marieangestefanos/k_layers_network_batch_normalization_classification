function preprocessed_X = Preprocess(X, mean, std)
    X = X - repmat(mean, [1, size(X, 2)]);
    preprocessed_X = X ./ repmat(std, [1, size(X, 2)]);
end
function [X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = ...
    SplitData(X, Y, y, validset_size)

    X_train = X(:, 1:end-validset_size);
    Y_train = Y(:, 1:end-validset_size);
    y_train = y(1:end-validset_size);
    
    X_valid = X(:, end-validset_size+1:end);
    Y_valid = Y(:, end-validset_size+1:end);
    y_valid = y(end-validset_size+1:end);

end
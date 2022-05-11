function Grads = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams, lambda)
    
    n_batch = size(X_batch, 2);
    k = NetParams.k;
    W = NetParams.W;
    b = NetParams.b;
    grad_W = cell(1, k);
    grad_b = cell(1, k);
    
    % Propagate the gradient through the loss and softmax operations
    G_batch = - (Y_batch - P_batch);

    for l=k:-1:2
        % Compute grad of J wrt W{l} and b{l}
        grad_W{l} = (G_batch * Xs_batch{l-1}')/n_batch + 2 * lambda * W{l};
        grad_b{l} = (G_batch * ones(n_batch, 1))/n_batch;

        % Propagate G_batch to the previous layer
        G_batch = W{l}' * G_batch;
        G_batch( Xs_batch{l-1} <= 0 ) = 0;
    end

    grad_W{1} = (G_batch * X_batch')/n_batch + 2 * lambda * W{1};
    grad_b{1} = (G_batch * ones(n_batch, 1))/n_batch;

    Grads.W = grad_W;
    Grads.b = grad_b;
end
function Grads = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams, lambda, varargin)
    
    n_batch = size(X_batch, 2);
    k = NetParams.k;
    W = NetParams.W;
    b = NetParams.b;
    grad_W = cell(1, k);
    grad_b = cell(1, k);
    
    % Propagate the gradient through the loss and softmax operations
    G_batch = - (Y_batch - P_batch);

    if ~NetParams.use_bn %% no batch norm

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
    
    else %% with batch norm

        gammas = NetParams.gammas;
        betas = NetParams.betas;

        S = varargin{1};
        S_hat = varargin{2};
        mu = varargin{3};
        v = varargin{4};
        

        % Compute grad of J wrt W{k} and b{k}
        grad_W{k} = (G_batch * Xs_batch{k-1}')/n_batch + 2 * lambda * W{k};
        grad_b{k} = (G_batch * ones(n_batch, 1))/n_batch;
        
        % Propagate G_batch to the previous layer
        G_batch = W{k}' * G_batch;
        G_batch( Xs_batch{k-1} <= 0 ) = 0;
        
        Xs_all = [X_batch Xs_batch];
        grad_gammas = cell(1, k-1);
        grad_betas = cell(1, k-1);    

        for l=(k-1):-1:1

            % Compute gradient for the scale and offset parameters for layer l
            grad_gammas{l} = (G_batch .* S_hat{l}) * ones(n_batch, 1) / n_batch;
            grad_betas{l} = (G_batch * ones(n_batch, 1)) / n_batch;

            % Propagate the gradients through the scale and shift
            G_batch = G_batch .* (gammas{l} * ones(1, n_batch));
        
            % Propagate G_batch through the batch normalization
            G_batch = BatchNormBackPass(G_batch, S{l}, mu{l}, v{l});

            % The gradients of J wrt. bias vector b{l} and W{l}
            grad_W{l} = (G_batch * Xs_all{l}')/n_batch + 2 * lambda * W{l};
            grad_b{l} = (G_batch * ones(n_batch, 1))/n_batch;

            % If l > 1 propagate G_batch to the previous layer
            if l > 1
                G_batch = W{l}' * G_batch;
                G_batch( Xs_all{l} <= 0 ) = 0;
            end

        end

        Grads.gammas = grad_gammas;
        Grads.betas = grad_betas;

    end

    Grads.W = grad_W;
    Grads.b = grad_b;
end
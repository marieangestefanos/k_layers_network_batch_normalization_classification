function [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta, lambda)
    
    n_batch = size(X_batch, 2);
    
    G_batch = - (Y_batch - P_batch);
    
    grad_W2 = (G_batch * H_batch')/n_batch + 2 * lambda * theta{2};
    grad_b2 = (G_batch * ones(n_batch, 1))/n_batch;
    
    %Propagate the grad back through the second layer
    G_batch = theta{2}' * G_batch;
    G_batch( H_batch <= 0 ) = 0;
    
    grad_W1 = (G_batch * X_batch')/n_batch + 2 * lambda * theta{1};
    grad_b1 = (G_batch * ones(n_batch, 1))/n_batch;
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
end
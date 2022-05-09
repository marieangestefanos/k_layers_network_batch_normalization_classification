function [Wstar, bstar, J_train_array, loss_train_array, ...
    J_valid_array, loss_valid_array, acc_train, acc_valid] = ...
    MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, theta, lambda)

    n = size(X_train, 2);
    
    n_batch = GDparams{1};
    eta = GDparams{2};

    n_epochs = GDparams{3};
    
        J_train_array = zeros(1, n_epochs);
        loss_train_array = zeros(1, n_epochs);
        J_valid_array = zeros(1, n_epochs);
        loss_valid_array = zeros(1, n_epochs);
        acc_train = zeros(1, n_epochs);
        acc_valid = zeros(1, n_epochs);
    
        for epoch = 1:n_epochs
            for j=1:n/n_batch
                
                idx_permutation = Shuffle(1:n);
        
                j_start = (j-1)*n_batch + 1;
                j_end = j*n_batch;
                inds = j_start:j_end;
        
                X_batch = X_train(:, idx_permutation(inds));
                Y_batch = Y_train(:, idx_permutation(inds));
        
                [H_batch, P_batch] = EvaluateClassifier(X_batch, theta);
        
                [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta, lambda);
        
                theta{1} = theta{1} - eta * grad_W{1};
                theta{2} = theta{2} - eta * grad_W{2};
                theta{3} = theta{3} - eta * grad_b{1};
                theta{4} = theta{4} - eta * grad_b{2};
        
            end
        
            [J_train, loss_train] = ComputeCost(X_train, Y_train, theta(1:2), theta(3:4), lambda);
            [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, theta(1:2), theta(3:4), lambda);
        
            J_train_array(epoch) = J_train;
            loss_train_array(epoch) = loss_train;
            J_valid_array(epoch) = J_valid;
            loss_valid_array(epoch) = loss_valid;
    
            acc_train(epoch) = ComputeAccuracy(X_train, y_train, theta);
            acc_valid(epoch) = ComputeAccuracy(X_valid, y_valid, theta);
        
        end

    Wstar = theta(1:2);
    bstar = theta(3:4);

end
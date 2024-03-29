function [NetParams, J_train_array, loss_train_array, ...
    J_valid_array, loss_valid_array, acc_train, acc_valid, etas, mu_av, v_av] = ...
    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams)

    n = size(X_train, 2);
    k = NetParams.k;
    
    nb_cycles = etaparams{1};
    n_s = etaparams{2};
    eta_min = etaparams{3};
    eta_max = etaparams{4};
    
    J_train_array = [];
    loss_train_array = [];
    J_valid_array = [];
    loss_valid_array = [];
    acc_train = [];
    acc_valid = [];
    
    
    t = 0;
    epoch = 1;
    l = 0;
    
    etas = [];
    
    if ~NetParams.use_bn
        
        while l < nb_cycles
            
            for j=1:n/n_batch % = 1000 = 1 cycle
                
                
                idx_permutation = Shuffle(1:n);
                
                j_start = (j-1)*n_batch + 1;
                j_end = j*n_batch;
                inds = j_start:j_end;
                
                X_batch = X_train(:, idx_permutation(inds));
                Y_batch = Y_train(:, idx_permutation(inds));
        
                [Xs_batch, P_batch] = EvaluateClassifier(X_batch, NetParams);
                    
                Grads = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams, lambda);
                grad_W = Grads.W;
                grad_b = Grads.b;
                    
                % Cycling learning rate
                if mod(floor(t/n_s), 2) == 0 %case of positive slope
                    eta = eta_min + (t - 2*l*n_s)/n_s*(eta_max - eta_min);
                else %case of negative slope
                    eta = eta_max - (t - (2*l+1)*n_s)/n_s*(eta_max - eta_min);
                end
                
                for i=1:k
                    NetParams.W{i} = NetParams.W{i} - eta * Grads.W{i};
                    NetParams.b{i} = NetParams.b{i} - eta * Grads.b{i};
                end
                
                etas(end+1) = eta;
                t = t + 1;
                l = floor(t/(2*n_s));
                
                
                [J_train, loss_train] = ComputeCost(X_train, Y_train, NetParams, lambda);
                [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, NetParams, lambda);
                
                interval_size = n/n_batch/10; % 10 = wanted_nb_points_per_cycle
                
                if mod(t, interval_size) == 0
                    
                    J_train_array(end+1) = J_train;
                    loss_train_array(end+1) = loss_train;
                    J_valid_array(end+1) = J_valid;
                    loss_valid_array(end+1) = loss_valid;
                    
                    acc_train(end+1) = ComputeAccuracy(X_train, y_train, NetParams);
                    acc_valid(end+1) = ComputeAccuracy(X_valid, y_valid, NetParams);

                end
                
                if mod(t, 100) == 0    
                    t
                end
                
            end
            
            epoch = epoch + 1;
            
        end
        
        
    else %% batch norm
        
        alpha = NetParams.alpha;

        while l < nb_cycles
            
            for j=1:n/n_batch % = 1000 = 1 cycle
                
                
                idx_permutation = Shuffle(1:n);
                
                j_start = (j-1)*n_batch + 1;
                j_end = j*n_batch;
                inds = j_start:j_end;
                
                X_batch = X_train(:, idx_permutation(inds));
                Y_batch = Y_train(:, idx_permutation(inds));
                
                [Xs_batch, P_batch, S, S_hat, mu, v] = EvaluateClassifier(X_batch, NetParams);
                
                %% Exponential moving average
                if l == 0 & j == 1
                    mu_av = mu;
                    v_av = v;
                else
                    for layer=1:k-1
                        mu_av{layer} = alpha * mu_av{layer} + (1-alpha) * mu{layer};
                        v_av{layer} = alpha * v_av{layer} + (1-alpha) * v{layer};
                        % mu_av = cellfun(@(x) x*alpha, mu_av, 'un', 0) + cellfun(@(x) x*(1-alpha), mu, 'un', 0);
                        % v_av = cellfun(@(x) x*alpha, v_av, 'un', 0) + cellfun(@(x) x*(1-alpha), v, 'un', 0);
                    end
                end
                    
                Grads = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams, lambda, S, S_hat, mu, v);
                grad_W = Grads.W;
                grad_b = Grads.b;
                    
                % Cycling learning rate
                if mod(floor(t/n_s), 2) == 0 %case of positive slope
                    eta = eta_min + (t - 2*l*n_s)/n_s*(eta_max - eta_min);
                else %case of negative slope
                    eta = eta_max - (t - (2*l+1)*n_s)/n_s*(eta_max - eta_min);
                end

                for i=1:k
                    NetParams.W{i} = NetParams.W{i} - eta * Grads.W{i};
                    NetParams.b{i} = NetParams.b{i} - eta * Grads.b{i};
                end

                etas(end+1) = eta;
                t = t + 1;
                l = floor(t/(2*n_s));

                [J_train, loss_train] = ComputeCost(X_train, Y_train, NetParams, lambda, mu_av, v_av);
                [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, NetParams, lambda, mu_av, v_av);
                
                interval_size = n/n_batch/10; % 10 = wanted_nb_points_per_cycle

                if mod(t, interval_size) == 0

                    J_train_array(end+1) = J_train;
                    loss_train_array(end+1) = loss_train;
                    J_valid_array(end+1) = J_valid;
                    loss_valid_array(end+1) = loss_valid;
            
                    acc_train(end+1) = ComputeAccuracy(X_train, y_train, NetParams, mu_av, v_av);
                    acc_valid(end+1) = ComputeAccuracy(X_valid, y_valid, NetParams, mu_av, v_av);

                end

                if mod(t, 100) == 0    
                    t
                end
        
            end

            epoch = epoch + 1;
                        
        end
    end
end
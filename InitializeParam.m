function NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn)
    [d, ~] = size(X_train);
    [K, ~] = size(Y_train);
    
    nb_hid_layers = size(hid_dim, 2);

    if strcmp(init_type, "xavier")
        
        % First hidden layer param
        W{1} = randn( hid_dim(1), d ) / sqrt( d );
        b{1} = zeros(hid_dim(1), 1);
        
        % From second to last hidden layer param
        if nb_hid_layers > 1
            for i = 1:(nb_hid_layers-1)
                W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) / sqrt(hid_dim(i));
                b{i+1} = zeros(hid_dim(i+1), 1);
            end
        else
            i = 0;
        end

        % Output layer param
        W{end+1} = randn( K, hid_dim(i+1) ) / sqrt(hid_dim(i+1));
        b{end+1} = zeros(K, 1);
    
    elseif strcmp(init_type, "he")
        % First hidden layer param
        W{1} = randn( hid_dim(1), d ) * sqrt( 2 / d );
        b{1} = zeros(hid_dim(1), 1);
        
        % From second to last hidden layer param
        if nb_hid_layers > 1
            for i = 1:(nb_hid_layers-1)
                W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) * sqrt( 2 / hid_dim(i));
                b{i+1} = zeros(hid_dim(i+1), 1);
            end
        else
            i = 0;
        end

        % Output layer param
        W{end+1} = randn( K, hid_dim(i+1) ) * sqrt( 2 / hid_dim(i+1));
        b{end+1} = zeros(K, 1);
    
    else
        error('Error in InitializeParam: wrong init_type.')
    
    end
    
    NetParams.W = W;
    NetParams.b = b;
    NetParams.k = nb_hid_layers + 1;
    NetParams.use_bn = use_bn;
end
function NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, varargin)
    [d, ~] = size(X_train);
    [K, ~] = size(Y_train);
    
    nb_hid_layers = size(hid_dim, 2);

    if nargin < 7
        if strcmp(init_type, "xavier")
            
            % First hidden layer param
            W{1} = randn( hid_dim(1), d ) / sqrt( d );
            b{1} = zeros(hid_dim(1), 1);
            
            if use_bn
                gammas{1} = ones(hid_dim(1), 1);
                betas{1} = zeros(hid_dim(1), 1);
            end    
            
            % From second to last hidden layer param

                %% No batch norm
            if ~ use_bn

                if nb_hid_layers > 1
                    for i = 1:(nb_hid_layers-1)
                        W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) / sqrt(hid_dim(i));
                        b{i+1} = zeros(hid_dim(i+1), 1);
                    end
                else
                    i = 0;
                end

            else % use batch norm

                if nb_hid_layers > 1
                    for i = 1:(nb_hid_layers-1)
                        W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) / sqrt(hid_dim(i));
                        b{i+1} = zeros(hid_dim(i+1), 1);
                        gammas{i+1} = ones(hid_dim(i+1), 1);
                        betas{i+1} = zeros(hid_dim(i+1), 1);
                    end
                else
                    i = 0;
                end

            end

            % Output layer param
            W{end+1} = randn( K, hid_dim(i+1) ) / sqrt(hid_dim(i+1));
            b{end+1} = zeros(K, 1);
            if use_bn
                gammas{end+1} = ones(K, 1);
                betas{end+1} = zeros(K, 1);
            end
        
        elseif strcmp(init_type, "he")
            % First hidden layer param
            W{1} = randn( hid_dim(1), d ) * sqrt( 2 / d );
            b{1} = zeros(hid_dim(1), 1);

            if use_bn
                gammas{1} = ones(hid_dim(1), 1);
                betas{1} = zeros(hid_dim(1), 1);
            end
            
            % From second to last hidden layer param

            %% No batch norm
            if ~ use_bn

                if nb_hid_layers > 1
                    for i = 1:(nb_hid_layers-1)
                        W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) * sqrt( 2 / hid_dim(i));
                        b{i+1} = zeros(hid_dim(i+1), 1);
                    end
                else
                    i = 0;
                end

            else % use batch norm

                if nb_hid_layers > 1
                    for i = 1:(nb_hid_layers-1)
                        W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) * sqrt( 2 / hid_dim(i));
                        b{i+1} = zeros(hid_dim(i+1), 1);
                        gammas{i+1} = ones(hid_dim(i+1), 1);
                        betas{i+1} = zeros(hid_dim(i+1), 1);
                    end
                else
                    i = 0;
                end

            end

            % Output layer param
            W{end+1} = randn( K, hid_dim(i+1) ) * sqrt( 2 / hid_dim(i+1));
            b{end+1} = zeros(K, 1);

            if use_bn
                gammas{end+1} = ones(K, 1);
                betas{end+1} = zeros(K, 1);
            end
        
        else
            error('Error in InitializeParam: wrong init_type.')
        
        end
    
    else %%sensitivy to initialization, question v)
        
        sig = varargin{2};

        % First hidden layer param
        W{1} = randn( hid_dim(1), d ) * sqrt( 2 / sig );
        b{1} = zeros(hid_dim(1), 1);

        if use_bn
            gammas{1} = ones(hid_dim(1), 1);
            betas{1} = zeros(hid_dim(1), 1);
        end
        
        % From second to last hidden layer param

        %% No batch norm
        if ~ use_bn

            if nb_hid_layers > 1
                for i = 1:(nb_hid_layers-1)
                    W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) * sqrt( 2 / sig );
                    b{i+1} = zeros(hid_dim(i+1), 1);
                end
            else
                i = 0;
            end

        else % use batch norm

            if nb_hid_layers > 1
                for i = 1:(nb_hid_layers-1)
                    W{i+1} = randn( hid_dim(i+1), hid_dim(i) ) * sqrt( 2 / sig );
                    b{i+1} = zeros(hid_dim(i+1), 1);
                    gammas{i+1} = ones(hid_dim(i+1), 1);
                    betas{i+1} = zeros(hid_dim(i+1), 1);
                end
            else
                i = 0;
            end

        end

        % Output layer param
        W{end+1} = randn( K, hid_dim(i+1) ) * sqrt( 2 / sig );
        b{end+1} = zeros(K, 1);

        if use_bn
            gammas{end+1} = ones(K, 1);
            betas{end+1} = zeros(K, 1);
        end

    end

    NetParams.W = W;
    NetParams.b = b;
    NetParams.k = nb_hid_layers + 1;
    NetParams.use_bn = use_bn;
    
    if use_bn
        NetParams.alpha = varargin{1};
        NetParams.gammas = gammas;
        NetParams.betas = betas;
    end;
end
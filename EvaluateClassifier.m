function [Xs, P, S, S_hat, mu, v] = EvaluateClassifier(X, NetParams, varargin)
    W = NetParams.W;
    b = NetParams.b;
    k = NetParams.k;
    gamma = NetParams.gamma;
    beta = NetParams.beta;

    if ~(NetParams.use_bn) %%no batch norm

        mu = None;
        v = None;

        Xs{1} = max(0, W{1} * X + repmat(b{1}, [1, size(X, 2)]));

        for i=2:(k-1)
            Xs{end + 1} = max(0, W{i} * Xs{i-1} + repmat(b{i}, [1, size(Xs{i-1}, 2)]));
        end
    
        % k - 1 = Nb of calculated Xs, input not included
        if size(Xs, 2) ~= (k-1)
            error("Error in EvaluateClassifier: size(Xs, 2) must equal k-1")
        end
        
        % Final linear transformation
        s = W{k} * Xs{k-1} + repmat(b{k}, [1, size(X, 2)]);
        P = softmax(s);

    else %% batch norm

        n = size(X, 2);
        
        %% un-norm scores of layer 1
        S{1} = W{1} * X + repmat(b{1}, [1, n]);

        %% normalized scores of layer i
        if nargin == 4 %% testset
            mu = varargin{3};
            v = varargin{4};    
        else %% trainset or validset
            % s = S{1};
            % mean(ones(3, 4))
            mu{1} = mean(S{1}, 2);
            v{1} = var(S{1}, 0, 2) * (n-1) / n;
        end
        S_hat{1} = BatchNormalize(S{1}, mu{1}, v{1});

        %% scale and shift no need to save them for backward pass
        S_tilde = repmat(gamma{1}, [1, n]) .* S_hat{1} + repmat(beta{1}, [1, n]);

        % ReLu
        Xs{1} = max(0, S_tilde);

        for i=2:(k-1)

            n = size(Xs{i-1}, 2);
            
            %% un-norm scores of layer i
            S{end + 1} = W{i} * Xs{i-1} + repmat(b{i}, [1, n]); 
            
            %% normalized scores of layer i

            if nargin == 2 %% trainset or validset
                mu{end + 1} = mean(S{i}, 2);
                v{end + 1} = var(S{i}, 0, 2) * (n-1) / n;
            end
            
            S_hat{end + 1} = BatchNormalize(S{i}, mu{i}, v{i});
            
            %% scale and shift no need to save them for backward pass
            S_tilde = repmat(gamma{i}, [1, n]) .* S_hat{i} + repmat(beta{i}, [1, n]);
            
            %% ReLu
            Xs{end + 1} = max(0, S_tilde);
        end

        % k - 1 = Nb of calculated Xs, input not included
        if size(Xs, 2) ~= (k-1)
            error("Error in EvaluateClassifier: size(Xs, 2) must equal k-1")
        end
        
        % Final linear transformation
        s = W{k} * Xs{k-1} + repmat(b{k}, [1, size(X, 2)]);
        P = softmax(s);

    end
end
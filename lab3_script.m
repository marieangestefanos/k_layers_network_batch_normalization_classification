addpath("/home/stefanom/Documents/kth/dl/labs/lab3");
train_path = "data_batch_1.mat";
valid_path = "data_batch_2.mat";
test_path = "test_batch.mat";

%% LOAD DATA
[X_train, Y_train, y_train] = LoadBatch(train_path);
[X_valid, Y_valid, y_valid] = LoadBatch(valid_path);
[X_test, Y_test, y_test] = LoadBatch(test_path);


%% PREPROCESS DATA
rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);
X_test = Preprocess(X_test, mean_train, std_train);

[~, n] = size(X_train);

%% INIT NETWORK PARAM
k = 3;
hid_dim = [20 50];
NetParams = InitializeParam(X_train, Y_train, hid_dim, k);
W = NetParams.W;
b = NetParams.b;
d_batch = 10;
end_batch = 3;
h = 1e-5;
lambda = 0;
eps = 1e-6;
n_batch = 100;
eta = 0.001;
n_epochs = 200;

%% EVAL THE NETWORK FUNCTION

X_batch = X_train(1:d_batch, 1:end_batch);
W1_batch = W{1};
W{1} = W1_batch(:, 1:d_batch);

NetParams.W = W;
[Xs, P] = EvaluateClassifier(X_batch, NetParams);

% [Xs, P] = EvaluateClassifier(X_train, NetParams);

% %% COMPUTE THE COST FUNCTION
% [J, loss] = ComputeCost(X_train, Y_train, W, b, lambda);

% %% COMPUTE THE GRADIENTS
% X_batch = X_train(1:d_batch, 1:end_batch);
% Y_batch = Y_train(:, 1:end_batch);
% theta_batch = InitializeParam(X_batch, Y_batch, m);
% W_batch = theta_batch(1:2);
% b_batch = theta_batch(3:4);

% %% GRADIENTS COMPARISONS
% [H_batch, P_batch] = EvaluateClassifier(X_batch, theta_batch);
% [grad_W_an, grad_b_an] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta_batch, lambda);

% % [grad_b_num_fast, grad_W_num_fast] = ComputeGradsNum(X_batch, Y_batch, W_batch, b_batch, lambda, h);
% [grad_b_num_slow, grad_W_num_slow] = ComputeGradsNumSlow(X_batch, Y_batch, W_batch, b_batch, lambda, h);

% % [grad_W_err_fast, grad_b_err_fast] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_fast, grad_b_num_fast, eps);
% [grad_W_err_slow, grad_b_err_slow] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_slow, grad_b_num_slow, eps);
% % [grad_W_err_given, grad_b_err_given] = ComputeRelativeError(grad_W_num_slow, grad_b_num_slow, grad_W_num_fast, grad_b_num_fast, eps);
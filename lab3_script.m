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
k = 4;
hid_dim = [50, 30, 20];
use_bn = false;
NetParams = InitializeParam(X_train, Y_train, hid_dim, k, use_bn);
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

%% Testing for a batch with less dimensions
X_batch = X_train(1:d_batch, 1:end_batch);
Y_batch = Y_train(:, 1:end_batch);
NetParams_batch = InitializeParam(X_batch, Y_batch, hid_dim, k, use_bn);
[Xs_batch, P_batch] = EvaluateClassifier(X_batch, NetParams_batch);

%% Testing for the whole trainset
% [Xs, P] = EvaluateClassifier(X_train, NetParams);

%% COMPUTE THE COST FUNCTION
[J, loss] = ComputeCost(X_train, Y_train, NetParams, lambda);

%% COMPUTE THE GRADIENTS
GradsAn = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams_batch, lambda);
GradsNum = ComputeGradsNumSlow(X_batch, Y_batch, NetParams_batch, lambda, h);

%% GRADIENTS COMPARISONS
Errors = ComputeRelativeError(GradsAn, GradsNum, eps);

for i=1:k
    fprintf('LAYER %d:\n', i)
    fprintf('W: %.2d\n', max(Errors.W{i}, [], 'all'))
    fprintf('b: %.2d\n\n', max(Errors.b{i}, [], 'all'))
end

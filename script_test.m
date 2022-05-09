addpath("Documents/kth/dl/labs/lab2/");
train_path = "data_batch_1.mat";
valid_path = "data_batch_2.mat";
test_path = "test_batch.mat";

%%Load data
[X_train, Y_train, y_train] = LoadBatch(train_path);
[X_valid, Y_valid, y_valid] = LoadBatch(valid_path);
[X_test, Y_test, y_test] = LoadBatch(test_path);

%%Preprocess data
rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);
X_test = Preprocess(X_test, mean_train, std_train);

[~, n] = size(X_train);

%%Initialize parameters of the network
m = 50;
theta = InitializeParam(X_train, Y_train, m);
W = theta(1:2);
b = theta(3:4);
d_batch = 10;
end_batch = 3;
h = 1e-5;
lambda = 0;
eps = 1e-6;
n_batch = 100;
eta = 0.001;
n_epochs = 200;

%%Evaluate the network function
[H, P] = EvaluateClassifier(X_train, theta);

%%Compute the cost function
[J, loss] = ComputeCost(X_train, Y_train, W, b, lambda);

%%Compute the gradients
X_batch = X_train(1:d_batch, 1:end_batch);
Y_batch = Y_train(:, 1:end_batch);
theta_batch = InitializeParam(X_batch, Y_batch, m);
W_batch = theta_batch(1:2);
b_batch = theta_batch(3:4);

%%Gradients comparisons
[H_batch, P_batch] = EvaluateClassifier(X_batch, theta_batch);
[grad_W_an, grad_b_an] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta_batch, lambda);

[grad_b_num_fast, grad_W_num_fast] = ComputeGradsNum(X_batch, Y_batch, W_batch, b_batch, lambda, h);
[grad_b_num_slow, grad_W_num_slow] = ComputeGradsNumSlow(X_batch, Y_batch, W_batch, b_batch, lambda, h);

[grad_W_err_fast, grad_b_err_fast] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_fast, grad_b_num_fast, eps);
[grad_W_err_slow, grad_b_err_slow] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_slow, grad_b_num_slow, eps);
[grad_W_err_given, grad_b_err_given] = ComputeRelativeError(grad_W_num_slow, grad_b_num_slow, grad_W_num_fast, grad_b_num_fast, eps);

max(grad_W_err_fast{1}, [], 'all')
max(grad_W_err_fast{2}, [], 'all')
max(grad_b_err_fast{1}, [], 'all')
max(grad_b_err_fast{2}, [], 'all')

max(grad_W_err_slow{1}, [], 'all')
max(grad_W_err_slow{2}, [], 'all')
max(grad_b_err_slow{1}, [], 'all')
max(grad_b_err_slow{2}, [], 'all')

max(grad_W_err_given{1}, [], 'all')
max(grad_W_err_given{2}, [], 'all')
max(grad_b_err_given{1}, [], 'all')
max(grad_b_err_given{2}, [], 'all')

%% Exercise 2
% GDparams = {n_batch, eta, n_epochs};
% [Wstar, bstar, J_train_array, loss_train_array, ...
%     J_valid_array, loss_valid_array, acc_train, acc_valid] = ...
%     MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, theta, lambda);

% figure;
% plot(1:n_epochs, J_train_array, 1:n_epochs, J_valid_array, '--');
% title('Cost J over epochs');
% xlabel('epochs');
% ylabel('cost J');
% legend('Training','Validation');
% figure;
% plot(1:n_epochs, loss_train_array, 1:n_epochs, loss_valid_array, '--');
% title('Loss over epochs');
% xlabel('epochs');
% ylabel('loss');
% legend('Training','Validation');
% figure;
% plot(1:n_epochs, acc_train, 1:n_epochs, acc_valid, '--');
% title('Accuracy over epochs');
% xlabel('epochs');
% ylabel('accuracy');
% legend('Training','Validation');

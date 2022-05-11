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
k = 2;
hid_dim = [50];
use_bn = false;

NetParams = InitializeParam(X_train, Y_train, hid_dim, k, use_bn);
W = NetParams.W;
b = NetParams.b;
d_batch = 10;
end_batch = 3;

h = 1e-5; %given grad fn param

lambda = 0;
eps = 1e-6; %grad cmp param

n_batch = 100;
n_epochs = 200;

%% EVAL THE NETWORK FUNCTION

% %% Testing for a batch with less dimensions
% X_batch = X_train(1:d_batch, 1:end_batch);
% Y_batch = Y_train(:, 1:end_batch);
% NetParams_batch = InitializeParam(X_batch, Y_batch, hid_dim, k, use_bn);
% [Xs_batch, P_batch] = EvaluateClassifier(X_batch, NetParams_batch);

% % Testing for the whole trainset
% [Xs, P] = EvaluateClassifier(X_train, NetParams);

% %% COMPUTE THE COST FUNCTION
% [J, loss] = ComputeCost(X_train, Y_train, NetParams, lambda);

% %% COMPUTE THE GRADIENTS
% GradsAn = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams_batch, lambda);
% GradsNum = ComputeGradsNumSlow(X_batch, Y_batch, NetParams_batch, lambda, h);

%%% Exercise 1

%% GRADIENTS COMPARISONS
% Errors = ComputeRelativeError(GradsAn, GradsNum, eps);
% for i=1:k
%     fprintf('LAYER %d:\n', i)
%     fprintf('W: %.2d\n', max(Errors.W{i}, [], 'all'))
%     fprintf('b: %.2d\n\n', max(Errors.b{i}, [], 'all'))
% end


%% Exercise 2
% %% Replicate assignment 2 results
% lambda = 0.01;
% eta_min = 1e-5;
% eta_max = 1e-1;
% n_s = 500;
% nb_cycles = 1;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% [NetParams_star, J_train_array, loss_train_array, ...
%    J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);

% len_array = size(J_train_array, 2);
% figure;
% update_steps = (1:len_array)*10;
% nb_updates = size(J_train_array, 2);
% plot(update_steps, J_train_array, update_steps, J_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Cost J over updates');
% xlabel('update steps');
% ylabel('cost J');
% legend('Training','Validation');
% figure;
% plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');
% figure;
% plot(update_steps, acc_train, update_steps, acc_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Accuracy over updates');
% xlabel('update steps');
% ylabel('accuracy');
% legend('Training','Validation');

% nb_updates = size(etas, 2);
% figure;
% plot(1:nb_updates, etas);
% title('Eta values over updates');
% xlabel('update steps');
% ylabel('eta');

% %%
% %% Print test set accuracy
% fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star)*100)

% 3-layer net (50-50)
k = 3;
hid_dim = [50, 50];
use_bn = false;

n_batch = 100;
eta_min = 1e-5;
eta_max = 1e-1;
lambda = 0.005;
n_s = 5 * 45000 / n_batch; %2250
nb_cycles = 2;
etaparams = {nb_cycles, n_s, eta_min, eta_max};

[NetParams_star, J_train_array, loss_train_array, ...
   J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
   MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);

len_array = size(J_train_array, 2);
figure;
update_steps = (1:len_array)*10;
nb_updates = size(J_train_array, 2);
plot(update_steps, J_train_array, update_steps, J_valid_array, '--');
yl = ylim;
ylim([0, yl(2)]);
title('Cost J over updates');
xlabel('update steps');
ylabel('cost J');
legend('Training','Validation');
figure;
plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
yl = ylim;
ylim([0, yl(2)]);
title('Loss over updates');
xlabel('update steps');
ylabel('loss');
legend('Training','Validation');
figure;
plot(update_steps, acc_train, update_steps, acc_valid, '--');
yl = ylim;
ylim([0, yl(2)]);
title('Accuracy over updates');
xlabel('update steps');
ylabel('accuracy');
legend('Training','Validation');

nb_updates = size(etas, 2);
figure;
plot(1:nb_updates, etas);
title('Eta values over updates');
xlabel('update steps');
ylabel('eta');

%%
% Print test set accuracy
fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star)*100)
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
hid_dim = [50, 30];
init_type = "xavier";
use_bn = false;

NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn);
W = NetParams.W;
b = NetParams.b;
d_batch = 10;
end_batch = 3;

h = 1e-5; %given grad fn param

lambda = 0;
eps = 1e-6; %grad cmp param

n_batch = 100;
n_epochs = 200;

% %% EVAL THE NETWORK FUNCTION
% 
% %% Testing for a batch with less dimensions
% X_batch = X_train(1:d_batch, 1:end_batch);
% Y_batch = Y_train(:, 1:end_batch);
% NetParams_batch = InitializeParam(X_batch, Y_batch, hid_dim, init_type, use_bn);
% [Xs_batch, P_batch] = EvaluateClassifier(X_batch, NetParams_batch);
% 
% % Testing for the whole trainset
% [Xs, P] = EvaluateClassifier(X_train, NetParams);
% 
% %% COMPUTE THE COST FUNCTION
% [J, loss] = ComputeCost(X_train, Y_train, NetParams, lambda);
% 
% %% COMPUTE THE GRADIENTS
% GradsAn = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams_batch, lambda);
% GradsNum = ComputeGradsNumSlow(X_batch, Y_batch, NetParams_batch, lambda, h);
% 
% %% Exercise 1
% 
% % GRADIENTS COMPARISONS
% Errors = ComputeRelativeError(GradsAn, GradsNum, eps);
% for i=1:k
%     fprintf('LAYER %d:\n', i)
%     fprintf('W: %.2d\n', max(Errors.W{i}, [], 'all'))
%     fprintf('b: %.2d\n\n', max(Errors.b{i}, [], 'all'))
% end


%% Exercise 2
% %% STEP 1 - Replicate assignment 2 results
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


%% EXERCISE 2 Step 2 - Data loading, splitting and preprocessing
% path_batch1 = "data_batch_1.mat";
% path_batch2 = "data_batch_2.mat";
% path_batch3 = "data_batch_3.mat";
% path_batch4 = "data_batch_4.mat";
% path_batch5 = "data_batch_5.mat";

% [X1, Y1, y1] = LoadBatch(path_batch1);
% [X2, Y2, y2] = LoadBatch(path_batch2);
% [X3, Y3, y3] = LoadBatch(path_batch3);
% [X4, Y4, y4] = LoadBatch(path_batch4);
% [X5, Y5, y5] = LoadBatch(path_batch5);

% X = [X1 X2 X3 X4 X5];
% Y = [Y1 Y2 Y3 Y4 Y5];
% y = [y1; y2; y3; y4; y5];

% validset_size = 5000;
% [X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = SplitData(X, Y, y, validset_size);


% rng(400);
% mean_train = mean(X_train, 2);
% std_train = std(X_train, 0, 2);

% X_train = Preprocess(X_train, mean_train, std_train);
% X_valid = Preprocess(X_valid, mean_train, std_train);


% %% 3-layer net (50-50) - Xavier init
% hid_dim = [50, 50];
% k = hid_dim + 1;
% init_type = "xavier";
% use_bn = false;

% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn);

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
% % Print test set accuracy
% fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star)*100)

% %% 3-layer net (50-50) - He init
% hid_dim = [50, 50];
% k = hid_dim + 1;
% init_type = "he";
% use_bn = false;

% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn);

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
% % Print test set accuracy
% fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star)*100)

% %% 9-layer net (50-50) - He init
% hid_dim = [50, 30, 20, 20, 10, 10, 10, 10];
% k = hid_dim + 1;
% init_type = "he";
% use_bn = false;

% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn);

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

%% Batch normalization - tests
hid_dim = [50 50];
k = size(hid_dim, 2) + 1;
init_type = "he";
use_bn = true;

n_batch = 100;
eta_min = 1e-5;
eta_max = 1e-1;
lambda = 0.005;
n_s = 5;
nb_cycles = 1;
etaparams = {nb_cycles, n_s, eta_min, eta_max};
alpha = 0.7;

% %% Testing for a batch with less dimensions
% d_batch = 10;
% end_batch = 3;
% X_batch = X_train(1:d_batch, 1:end_batch);
% Y_batch = Y_train(:, 1:end_batch);
% NetParams = InitializeParam(X_batch, Y_batch, hid_dim, init_type, use_bn, alpha);
% [Xs, P, S, S_hat, mu, v] = EvaluateClassifier(X_batch, NetParams);

% %% COMPUTE THE COST FUNCTION
% [J, loss] = ComputeCost(Y_batch, Y_batch, NetParams, lambda);

% %% COMPUTE THE GRADIENTS
% GradsAn = ComputeGradients(X_batch, Y_batch, Xs, P, NetParams, lambda, S, S_hat, mu, v);
% GradsNum = ComputeGradsNumSlow(X_batch, Y_batch, NetParams, lambda, h);

% % GRADIENTS COMPARISONS
% Errors = ComputeRelativeError(GradsAn, GradsNum, eps);
% for i=1:(k-1)
%     fprintf('LAYER %d:\n', i)
%     fprintf('W: %.2d\n', max(Errors.W{i}, [], 'all'))
%     fprintf('b: %.2d\n', max(Errors.b{i}, [], 'all'))
%     fprintf('gammas: %.2d\n', max(Errors.gammas{i}, [], 'all'))
%     fprintf('betas: %.2d\n\n', max(Errors.betas{i}, [], 'all'))
% end
% fprintf('LAYER %d:\n', k)
% fprintf('W: %.2d\n', max(Errors.W{k}, [], 'all'))
% fprintf('b: %.2d\n\n', max(Errors.b{k}, [], 'all'))

%% Exercise 3: 3- and 9-layer net (50-50) - He init

    %% Preprocess data
path_batch1 = "data_batch_1.mat";
path_batch2 = "data_batch_2.mat";
path_batch3 = "data_batch_3.mat";
path_batch4 = "data_batch_4.mat";
path_batch5 = "data_batch_5.mat";

[X1, Y1, y1] = LoadBatch(path_batch1);
[X2, Y2, y2] = LoadBatch(path_batch2);
[X3, Y3, y3] = LoadBatch(path_batch3);
[X4, Y4, y4] = LoadBatch(path_batch4);
[X5, Y5, y5] = LoadBatch(path_batch5);

X = [X1 X2 X3 X4 X5];
Y = [Y1 Y2 Y3 Y4 Y5];
y = [y1; y2; y3; y4; y5];

validset_size = 5000;
[X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = SplitData(X, Y, y, validset_size);

rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);

    %% Network settings

% % hid_dim = [50, 50];
% hid_dim = [50, 30, 20, 20, 10, 10, 10, 10];
% k = hid_dim + 1;
% init_type = "he";
% use_bn = true;

% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, alpha);

% [NetParams_star, J_train_array, loss_train_array, ...   
%    J_valid_array, loss_valid_array, acc_train, acc_valid, etas, mu_av, v_av] = ...
%    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);

% %%
% title_string = "/home/stefanom/Documents/kth/dl/labs/lab3/img/ex3_3lay_";
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
% saveas(gcf, title_string + "J.png");
% figure;
% plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');
% saveas(gcf, title_string + "loss.png")
% figure;
% plot(update_steps, acc_train, update_steps, acc_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Accuracy over updates');
% xlabel('update steps');
% ylabel('accuracy');
% legend('Training','Validation');
% saveas(gcf, title_string + "acc.png")

% %%
% nb_updates = size(etas, 2);
% figure;
% plot(1:nb_updates, etas);
% title('Eta values over updates');
% xlabel('update steps');
% ylabel('eta');
% saveas(gcf, title_string + "etas.png")

% %%
% % Print test set accuracy
% fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star, mu_av, v_av)*100)

%% Lambda search - coarch search 1

% save_path1 = "~/Documents/kth/dl/labs/lab3/Saved_Files/lambdas_results1.txt";

% hid_dim = [50, 50];
% k = hid_dim + 1;
% use_bn = true;

% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};

% nb_lambdas = 8;
% lambdas = linspace(1e-5, 1e-1, nb_lambdas);

% writecell({'lambda', 'acc_valid', 'acc_test'}, save_path1, 'Delimiter', 'tab');

% for lbda_idx=1:nb_lambdas

%     lambda = lambdas(lbda_idx)

%     NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, alpha);
    
%     [NetParams_star, J_train_array, loss_train_array, ...   
%     J_valid_array, loss_valid_array, acc_train, acc_valid, etas, mu_av, v_av] = ...
%     MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);

%     acc_valid(end)
%     test_acc = ComputeAccuracy(X_test, y_test, NetParams_star, mu_av, v_av)
    
%     writematrix([lambda acc_valid(end) test_acc],save_path1,'WriteMode','append', 'Delimiter', 'tab');

% end

%% Lambda search - fine search 1

% lambdas_results1 = importdata(save_path1);
% lambda_accuracies = lambdas_results1.data
% 
% path1 = "~/Documents/kth/dl/labs/lab3/Saved_Files/lambdas_results1.txt";
% 
% lbdas1 = importdata(path1);
% data1 = lbdas1.data;    
% [~, argmax] = max(data1(:, 2));
% lbda_min = data1(argmax - 1, 1);
% lbda_max = data1(argmax + 1, 1);

% lbda_min = 1e-5;
% lbda_max = 0.0285785714285714;
% nb_lambdas = 8;
% lambdas = linspace(lbda_min, lbda_max, nb_lambdas);
%  
% save_path2 = "~/Documents/kth/dl/labs/lab3/Saved_Files/lambdas_results2.txt";
% writecell({'lambda', 'acc_valid'}, save_path2, 'Delimiter', 'tab');
%  
% %% Recall settings
% hid_dim = [50, 50];
% k = hid_dim + 1;
% use_bn = true;
% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% writecell({'lambda', 'acc_valid', 'acc_test'}, save_path2, 'Delimiter', 'tab');
% 
% for lbda_idx=1:nb_lambdas
% 
%     lambda = lambdas(lbda_idx)
% 
%     NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, alpha);
%     
%     [NetParams_star, J_train_array, loss_train_array, ...   
%     J_valid_array, loss_valid_array, acc_train, acc_valid, etas, mu_av, v_av] = ...
%     MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);
% 
%     acc_valid(end)
%     test_acc = ComputeAccuracy(X_test, y_test, NetParams_star, mu_av, v_av)
%     
%     writematrix([lambda acc_valid(end) test_acc], save_path2, 'WriteMode', 'append', 'Delimiter', 'tab');
% 
% end


% %%% Sensitivity to initialization - With Batch Norm

rng(300);

% hid_dim = [50, 50];
% k = hid_dim + 1;
% use_bn = true;
% 
% n_batch = 100;
% eta_min = 1e-5;
% eta_max = 1e-1;
% lambda = 0.005;
% n_s = 5 * 45000 / n_batch; %2250
% nb_cycles = 2;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% sigmas = [1e-1 1e-2 1e-3 1e-4];
% 
% for sig_idx=1:length(sigmas)
% 
%     sig = sigmas(sig_idx);
%     NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, alpha, sig);
% 
%     [NetParams_star, J_train_array, loss_train_array, ...   
%     J_valid_array, loss_valid_array, acc_train, acc_valid, etas, mu_av, v_av] = ...
%     MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);
% 
%     title_string = "/home/stefanom/Documents/kth/dl/labs/lab3/img/qu5_BN_sig" + num2str(sig);
%     len_array = size(J_train_array, 2);
%     figure;
%     update_steps = (1:len_array)*10;
%     nb_updates = size(J_train_array, 2);
%     plot(update_steps, J_train_array, update_steps, J_valid_array, '--');
%     yl = ylim;
%     ylim([0, yl(2)]);
%     title('Cost J over updates');
%     xlabel('update steps');
%     ylabel('cost J');
%     legend('Training','Validation');
%     saveas(gcf, title_string + "J.png");
%     figure;
%     plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
%     yl = ylim;
%     ylim([0, yl(2)]);
%     title('Loss over updates');
%     xlabel('update steps');
%     ylabel('loss');
%     legend('Training','Validation');
%     saveas(gcf, title_string + "loss.png")
%     figure;
%     plot(update_steps, acc_train, update_steps, acc_valid, '--');
%     yl = ylim;
%     ylim([0, yl(2)]);
%     title('Accuracy over updates');
%     xlabel('update steps');
%     ylabel('accuracy');
%     legend('Training','Validation');
%     saveas(gcf, title_string + "acc.png")
% 
%     nb_updates = size(etas, 2);
%     figure;
%     plot(1:nb_updates, etas);
%     title('Eta values over updates');
%     xlabel('update steps');
%     ylabel('eta');
%     saveas(gcf, title_string + "etas.png")
% 
%     % Print test set accuracy
%     fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star, mu_av, v_av)*100)
% 
% end

%%% Sensitivity to initialization - Without Batch Norm

use_bn = false;

n_s = 5 * 45000 / n_batch; %2250
nb_cycles = 2;
etaparams = {nb_cycles, n_s, eta_min, eta_max};

sigmas = [1e-1 1e-2 1e-3];

for sig_idx=1:length(sigmas)

    sig = sigmas(sig_idx);
    NetParams = InitializeParam(X_train, Y_train, hid_dim, init_type, use_bn, alpha, sig);

    [NetParams_star, J_train_array, loss_train_array, ...   
    J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, NetParams, lambda, etaparams);

    title_string = "/home/stefanom/Documents/kth/dl/labs/lab3/img/qu5_noBN_sig" + num2str(sig);
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
    saveas(gcf, title_string + "J.png");
    figure;
    plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
    yl = ylim;
    ylim([0, yl(2)]);
    title('Loss over updates');
    xlabel('update steps');
    ylabel('loss');
    legend('Training','Validation');
    saveas(gcf, title_string + "loss.png")
    figure;
    plot(update_steps, acc_train, update_steps, acc_valid, '--');
    yl = ylim;
    ylim([0, yl(2)]);
    title('Accuracy over updates');
    xlabel('update steps');
    ylabel('accuracy');
    legend('Training','Validation');
    saveas(gcf, title_string + "acc.png")

    nb_updates = size(etas, 2);
    figure;
    plot(1:nb_updates, etas);
    title('Eta values over updates');
    xlabel('update steps');
    ylabel('eta');
    saveas(gcf, title_string + "etas.png")

    % Print test set accuracy
    fprintf('Test set accuracy: %.1f%%\n', ComputeAccuracy(X_test, y_test, NetParams_star, mu_av, v_av)*100)

end

%% FONCTIONS

% Compute argmax of each column
function argmax = Argmax(matrix)
    [~, argmax] = max(matrix);
end


function S_hat = BatchNormalize(S, mu, v)
    S_hat = diag( ( v + eps ).^(-1/2) ) * (S - mu);
end


function G_batch = BatchNormBackPass(G_batch, Sl_batch, mu_l, v_l)

    n_batch = size(G_batch, 2);

    sigma1 = (v_l + eps).^(-0.5);
    sigma2 = (v_l + eps).^(-1.5);
    G1 = G_batch .* (sigma1 * ones(1, n_batch));
    G2 = G_batch .* (sigma2 * ones(1, n_batch));
    D = Sl_batch - mu_l*ones(1, n_batch);
    c = (G2 .* D) * ones(n_batch, 1);

    G_batch = G1 - (G1 * ones(n_batch, 1)) * ones(1, n_batch) / n_batch - D .* (c * ones(1, n_batch)) / n_batch;

end


function accuracy = ComputeAccuracy(X, y, NetParams, varargin)
    n = size(y, 1);

    acc = zeros(1, n);

    if nargin < 4
        [~, P] = EvaluateClassifier(X, NetParams);
    else
        [~, P] = EvaluateClassifier(X, NetParams, varargin{1}, varargin{2});
    end

    prediction = Argmax(P);

    for i = 1:n
        if prediction(i) == y(i)
            acc(i) = 1;
        else
            acc(i) = 0;
        end
    end

    accuracy = sum(acc)/n;
end


function [J, loss] = ComputeCost(X, Y, NetParams, lambda, varargin)

    if nargin < 5
        [~, P] = EvaluateClassifier(X, NetParams);
    else
        [~, P] = EvaluateClassifier(X, NetParams, varargin{1}, varargin{2});
    end
    n = size(Y, 2);
    lcross = zeros(1, n);
    W = NetParams.W;

    for i = 1:n
        lcross(i) = Y(:, i)' * log(P(:, i));
    end
    loss = - sum(lcross)/n;

    sumW = 0;
    for i=1:NetParams.k
        sumW = sumW + sum(W{i} .* W{i}, 'all');
    end

    J = loss + lambda * sumW;
end


function Grads = ComputeGradients(X_batch, Y_batch, Xs_batch, P_batch, NetParams, lambda, varargin)
    
    n_batch = size(X_batch, 2);
    k = NetParams.k;
    W = NetParams.W;
    b = NetParams.b;
    grad_W = cell(1, k);
    grad_b = cell(1, k);
    
    % Propagate the gradient through the loss and softmax operations
    G_batch = - (Y_batch - P_batch);

    if ~NetParams.use_bn %% no batch norm

        for l=k:-1:2
            % Compute grad of J wrt W{l} and b{l}
            grad_W{l} = (G_batch * Xs_batch{l-1}')/n_batch + 2 * lambda * W{l};
            grad_b{l} = (G_batch * ones(n_batch, 1))/n_batch;

            % Propagate G_batch to the previous layer
            G_batch = W{l}' * G_batch;
            G_batch( Xs_batch{l-1} <= 0 ) = 0;
        end

        grad_W{1} = (G_batch * X_batch')/n_batch + 2 * lambda * W{1};
        grad_b{1} = (G_batch * ones(n_batch, 1))/n_batch;
    
    else %% with batch norm

        gammas = NetParams.gammas;
        betas = NetParams.betas;

        S = varargin{1};
        S_hat = varargin{2};
        mu = varargin{3};
        v = varargin{4};
        

        % Compute grad of J wrt W{k} and b{k}
        grad_W{k} = (G_batch * Xs_batch{k-1}')/n_batch + 2 * lambda * W{k};
        grad_b{k} = (G_batch * ones(n_batch, 1))/n_batch;
        
        % Propagate G_batch to the previous layer
        G_batch = W{k}' * G_batch;
        G_batch( Xs_batch{k-1} <= 0 ) = 0;
        
        Xs_all = [X_batch Xs_batch];
        grad_gammas = cell(1, k-1);
        grad_betas = cell(1, k-1);    

        for l=(k-1):-1:1

            % Compute gradient for the scale and offset parameters for layer l
            grad_gammas{l} = (G_batch .* S_hat{l}) * ones(n_batch, 1) / n_batch;
            grad_betas{l} = (G_batch * ones(n_batch, 1)) / n_batch;

            % Propagate the gradients through the scale and shift
            G_batch = G_batch .* (gammas{l} * ones(1, n_batch));
        
            % Propagate G_batch through the batch normalization
            G_batch = BatchNormBackPass(G_batch, S{l}, mu{l}, v{l});

            % The gradients of J wrt. bias vector b{l} and W{l}
            grad_W{l} = (G_batch * Xs_all{l}')/n_batch + 2 * lambda * W{l};
            grad_b{l} = (G_batch * ones(n_batch, 1))/n_batch;

            % If l > 1 propagate G_batch to the previous layer
            if l > 1
                G_batch = W{l}' * G_batch;
                G_batch( Xs_all{l} <= 0 ) = 0;
            end

        end

        Grads.gammas = grad_gammas;
        Grads.betas = grad_betas;

    end

    Grads.W = grad_W;
    Grads.b = grad_b;
end


function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

    Grads.W = cell(numel(NetParams.W), 1);
    Grads.b = cell(numel(NetParams.b), 1);
    if NetParams.use_bn
        Grads.gammas = cell(numel(NetParams.gammas), 1);
        Grads.betas = cell(numel(NetParams.betas), 1);
    end
    
    for j=1:length(NetParams.b)
        Grads.b{j} = zeros(size(NetParams.b{j}));
        NetTry = NetParams;
        for i=1:length(NetParams.b{j})
            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) - h;
            NetTry.b = b_try;
            c1 = ComputeCost(X, Y, NetTry, lambda);        
            
            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) + h;
            NetTry.b = b_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.b{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.W)
        Grads.W{j} = zeros(size(NetParams.W{j}));
            NetTry = NetParams;
        for i=1:numel(NetParams.W{j})
            
            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) - h;
            NetTry.W = W_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
        
            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) + h;
            NetTry.W = W_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
        
            Grads.W{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    if NetParams.use_bn
        for j=1:length(NetParams.gammas)
            Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.gammas{j})
                
                gammas_try = NetParams.gammas;
                gammas_try{j}(i) = gammas_try{j}(i) - h;
                NetTry.gammas = gammas_try;        
                c1 = ComputeCost(X, Y, NetTry, lambda);
                
                gammas_try = NetParams.gammas;
                gammas_try{j}(i) = gammas_try{j}(i) + h;
                NetTry.gammas = gammas_try;        
                c2 = ComputeCost(X, Y, NetTry, lambda);
                
                Grads.gammas{j}(i) = (c2-c1) / (2*h);
            end
        end
        
        for j=1:length(NetParams.betas)
            Grads.betas{j} = zeros(size(NetParams.betas{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.betas{j})
                
                betas_try = NetParams.betas;
                betas_try{j}(i) = betas_try{j}(i) - h;
                NetTry.betas = betas_try;        
                c1 = ComputeCost(X, Y, NetTry, lambda);
                
                betas_try = NetParams.betas;
                betas_try{j}(i) = betas_try{j}(i) + h;
                NetTry.betas = betas_try;        
                c2 = ComputeCost(X, Y, NetTry, lambda);
                
                Grads.betas{j}(i) = (c2-c1) / (2*h);
            end
        end    
    end


    function Errors = ComputeRelativeError(GradsAn, GradsNum, eps)

        k = size(GradsAn.W, 2); %nb of layers
        err_W = cell(1, k);
        err_b = cell(1, k);
        err_gammas = cell(1, k-1);
        err_betas = cell(1, k-1);
    
        %% Standford's course formula
        for i=1:(k-1)
            err_W{i} = abs(GradsAn.W{i} - GradsNum.W{i})./max(abs(GradsAn.W{i}), abs(GradsNum.W{i}));
            % err_b{i} = abs(GradsAn.b{i} - GradsNum.b{i})./max(abs(GradsAn.b{i}), abs(GradsNum.b{i}));
            % err_b{i} = abs(GradsAn.b{i} - GradsNum.b{i})./abs(GradsNum.b{i}); % Wikipedia formula
            err_b{i} = abs(GradsAn.b{i} - GradsNum.b{i})./max(eps, abs(GradsAn.b{i}) + GradsNum.b{i}); % Lab1 instructions formula
            err_gammas{i} = abs(GradsAn.gammas{i} - GradsNum.gammas{i})./max(abs(GradsAn.gammas{i}), abs(GradsNum.gammas{i}));
            err_betas{i} = abs(GradsAn.betas{i} - GradsNum.betas{i})./max(abs(GradsAn.betas{i}), abs(GradsNum.betas{i}));
        end
    
        err_W{k} = abs(GradsAn.W{k} - GradsNum.W{k})./max(abs(GradsAn.W{k}), abs(GradsNum.W{k}));
        % err_b{k} = abs(GradsAn.b{k} - GradsNum.b{k})./max(abs(GradsAn.b{k}), abs(GradsNum.b{k}));
        % err_b{k} = abs(GradsAn.b{k} - GradsNum.b{k})./abs(GradsNum.b{k}); % Wikipedia formula
        err_b{k} = abs(GradsAn.b{k} - GradsNum.b{k})./max(eps, abs(GradsAn.b{k}) + GradsNum.b{k}); % Lab1 instructions formula
    
    
        Errors.W = err_W;
        Errors.b = err_b;
        Errors.gammas= err_gammas;
        Errors.betas = err_betas;
    
    end


function [Xs, P, S, S_hat, mu, v] = EvaluateClassifier(X, NetParams, varargin)
    W = NetParams.W;
    b = NetParams.b;
    k = NetParams.k;
    
    if ~(NetParams.use_bn) %%no batch norm
        
        mu = -1;
        v = -1;
        
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
        
        gammas = NetParams.gammas;
        betas = NetParams.betas;

        n = size(X, 2);
        
        %% un-norm scores of layer 1
        S{1} = W{1} * X + repmat(b{1}, [1, n]);
        
        %% normalized scores of layer i
        if nargin > 2 %% testset or compute acc or cost durint training
            mu = varargin{1};
            v = varargin{2};
        else %% during training
            mu{1} = mean(S{1}, 2);
            v{1} = var(S{1}, 0, 2) * (n-1) / n;
        end
        S_hat{1} = BatchNormalize(S{1}, mu{1}, v{1});

        %% scale and shift no need to save them for backward pass
        S_tilde = repmat(gammas{1}, [1, n]) .* S_hat{1} + repmat(betas{1}, [1, n]);

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
            S_tilde = repmat(gammas{i}, [1, n]) .* S_hat{i} + repmat(betas{i}, [1, n]);
            
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


function [X, Y, y] = LoadBatch(filename)
    dict = load(filename);
    X = double(dict.data');
    y = dict.labels + 1;
    Y = (y == 1:10)';
end


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


function preprocessed_X = Preprocess(X, mean, std)
    X = X - repmat(mean, [1, size(X, 2)]);
    preprocessed_X = X ./ repmat(std, [1, size(X, 2)]);
end


function permutation = Shuffle(vector)
    permutation = vector(randperm(length(vector)));
end


function P = softmax(s)
    K = size(s, 1);
    P = exp(s)./repmat(ones(1, K)*exp(s), [K, 1]);
end


function [X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = ...
    SplitData(X, Y, y, validset_size)

    X_train = X(:, 1:end-validset_size);
    Y_train = Y(:, 1:end-validset_size);
    y_train = y(1:end-validset_size);
    
    X_valid = X(:, end-validset_size+1:end);
    Y_valid = Y(:, end-validset_size+1:end);
    y_valid = y(end-validset_size+1:end);

end
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
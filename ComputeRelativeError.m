function Errors = ComputeRelativeError(GradsAn, GradsNum, eps)

    k = size(GradsAn.W, 2); %nb of layers
    err_W = cell(1, k);
    err_b = cell(1, k);

    %% Standford's course formula
    for i=1:k
        size(GradsAn.W{i})
        size(GradsNum.W{i})
        err_W{i} = abs(GradsAn.W{i} - GradsNum.W{i})./max(abs(GradsAn.W{i}), abs(GradsNum.W{i}));
        err_b{i} = abs(GradsAn.b{i} - GradsNum.b{i})./max(abs(GradsAn.b{i}), abs(GradsNum.b{i}));
    end

    Errors.W = err_W;
    Errors.b = err_b;

end
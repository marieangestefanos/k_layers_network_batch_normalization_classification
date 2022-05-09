function [grad_W_err, grad_b_err] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num, grad_b_num, eps)
    %% Instructions formula
    %grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./max(eps, abs(grad_W_an{1}) + grad_W_num{1});
    %grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./max(eps, abs(grad_W_an{2}) + grad_W_num{2});
    %grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./max(eps, abs(grad_b_an{1}) + grad_b_num{1});
    %grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./max(eps, abs(grad_b_an{2}) + grad_b_num{2});
    
    %% Wikipedia formula
    %grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./abs(grad_W_num{1});
    %grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./abs(grad_W_num{2});
    %grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./abs(grad_b_num{1});
    %grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./abs(grad_b_num{2});
    
    %% Standford's course formula
    grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./max(abs(grad_W_num{1}), abs(grad_W_an{1}));
    grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./max(abs(grad_W_num{2}), abs(grad_W_an{2}));
    grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./max(abs(grad_b_num{1}), abs(grad_b_an{1}));
    grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./max(abs(grad_b_num{2}), abs(grad_b_an{2}));
    
    grad_W_err = {grad_W1_err, grad_W2_err};
    grad_b_err = {grad_b1_err, grad_b2_err};
end
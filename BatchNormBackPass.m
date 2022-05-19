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
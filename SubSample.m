function subsampled = SubSample(array, wanted_nbpoints_per_cycle, ...
    actual_nbpoints_per_cycle, nb_points_tot)
    
    interval_size = actual_nbpoints_per_cycle/wanted_nbpoints_per_cycle;
    sub_idx = interval_size * (1:(nb_points_tot/interval_size));
    subsampled = array(sub2ind(size(array), ones(1, size(sub_idx, 2)), sub_idx));

end
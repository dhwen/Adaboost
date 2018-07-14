function err_weights = get_err_weights(train_images,train_labels,arr_J,arr_T,arr_step_size,cur_iter,c)
    err_weights = zeros(size(train_labels,1),1); %// the weight of misclassifying each training input
     
    for i = 1:size(train_labels,1)
        if (train_labels(i)==(c-1))
            y = 1;
        else
            y = -1;
        end
        
        g = 0;
        for iter = 1:cur_iter
            g = g + arr_step_size(c,iter) * sign_dw(train_images(i,arr_J(c,iter)) - arr_T(c,iter)); 
        end
        err_weights(i) = exp(-y*g);
    end
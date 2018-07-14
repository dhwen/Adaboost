function [alpha_t,j_best,t_best] = getalphas(C,train_image,train_label,err_weights)

    train_len = size(train_image,1);
    input_len = size(train_image,2);
    N = 50;
    labels = -ones(size(train_image,1),1);
    class_indices = find(train_label == C-1);
    labels(class_indices) = 1;
    alpha_t = zeros(train_len,1);
    
    ones_mat = ones(train_len,input_len);
    ones_vec = ones(1,input_len);
    us = ones(size(ones_mat));
    
    for j=0:N
        indices = find(train_image(:,:) - j/N*ones_mat < 0);
        us(indices) = -1;
        alphas = (labels * ones_vec) .* us .* (err_weights * ones_vec);
        sum_alphas = sum(alphas,1);

        alphas_test = sign_dw(max(sum_alphas)-max(-sum_alphas))*alphas;
        
        [max_val, max_index] = max(sum(alphas_test,1));
        
        if ((max_val > sum(alpha_t))||(j==0))
            j_best = max_index;
            t_best = j/N;
            alpha_t = alphas_test(:,j_best);
        end
    end       
end
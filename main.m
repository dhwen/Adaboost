clear all;
close all;
clc;

Train_imgFile = 'trainingset\train-images.idx3-ubyte';
Train_labelFile = 'trainingset\train-labels.idx1-ubyte';

Test_imgFile = 'testset\t10k-images.idx3-ubyte';
Test_labelFile = 'testset\t10k-labels.idx1-ubyte';

%//Directly load data in .mat format instead of calling readMNIST
%[train_images train_labels] = readMNIST(Train_imgFile, Train_labelFile, 20000, 0);
%[test_images test_labels] = readMNIST(Test_imgFile, Test_labelFile, 10000, 0);
load('ReadData.mat');

C = 10;
xSize = 784;
TrainLength = size(train_images,1);
TestLength = size(test_images,1);
iter_max = 100;
iter = 0;
 
%//At each iteration, we find an input index j and decision stump t which has
% the best performance in predicting the correct class labels at the current time 
% via the following weak learner classifier: (borrowing C++ notation) y = im(j) > t ? 1 : -1;
% These weak learners are aggregated across iterations to construct the Adaboost classifier.

arr_J = ones(C,iter_max); %//Array of decision indices per class C at iteration i
arr_T = ones(C,iter_max); %//Array of decision stump thresholds per class C at iteration i
arr_step_size = ones(C,iter_max); %//Array of step sizes per class C at iteration i
arr_error_avg = ones(iter_max,1);

while(iter < iter_max)
    cum_err = 0;
    iter = iter + 1
    
    for c = 1:C    
        err_weights = get_err_weights(train_images,train_labels,arr_J,arr_T,arr_step_size,iter,c);
        [alpha,j,t]= getalphas(c,train_images,train_labels,err_weights);   
        misclassified_indices = find(sign_dw(alpha) == -1);     
        err = sum(err_weights(misclassified_indices)) / sum(err_weights);
        step_size = 1/2 * log((1 - err) / err);
        
        arr_J(c,iter) = j;
        arr_T(c,iter) = t;
        arr_step_size(c,iter) = step_size;
        cum_err = cum_err + err;
    end
    
	cum_err / C
    arr_error_avg(iter) = cum_err / C;
    
end

plot(1:iter_max, arr_error_avg)
xlabel 'Iteration', ylabel 'Average Error across Classes'
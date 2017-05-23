function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1 : (hidden_layer_size * (input_layer_size + 1))), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

  %% 1. expand y output to matrix
    y_matrix= eye(num_labels)(y,:); %result dimension 5000*10
  
  %% 2. perform forward propagation
    a1 = [ones(1,m); X'];               %add 1 column. Therefore, output has 5000 * 401 dimension 
    z2 = Theta1*a1;                      %Theta1 is 25 * 401; z2 is 25*5000
    
    a2 = sigmoid(z2);
    a2 = [ones(1,m) ;  a2];              %add 1 bias unit. Therefore, output has 26 * 5000 dimension 
    z3 = Theta2*a2;                       %Theta2 is 10 * 26; z3  is 10 *5000
    
    a3 = sigmoid(z3);
    
    h = a3' ;                                  %result dimension is 5000*10;    
    %%[y_pred, p_ind] = max(h, [], 2);           % find the max for each row.    
    
   %% 3. compute cost
   sum_k = sum((-y_matrix .*log(h))-((1-y_matrix).*log(1-h)),2);
    J = (1/m) * sum(sum_k);
   
   %% 4. compute regularized components; regulization = lambda/2m * sum of square theta_j
    sum_theta1_sqr = sum(sum(Theta1(:,2:end).^2, 2));
    sum_theta2_sqr = sum(sum(Theta2(:,2:end).^2, 2));
    reg = (lambda/(2*m)) * (sum_theta1_sqr + sum_theta2_sqr);
    
    J = J + reg;
    
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
    %% step 7. calculate backpropagation
     %   delta_2 = 0;

     %   d_3 = a3 - y';                                          % 10*5000 (or y_matrix?)
    
     %   d_2 = Theta2' * d_3 * sigmoidGradient(z2);        % 26*5000
    
     %   d_2 = d_2(2:end,:);                                        % exclude bias unit
    
     %   delta_2 = delta_2 + (d_3*a_2');
    
    %Approach2: loop
    delta_2 = zeros(size(Theta1));          %25 * 401
    delta_3 = zeros(size(Theta2));          %10 *26
    
    
    for t=1:m
        d_3 = a3(:,t) - y_matrix(t,:)';                           %10*1
        d_2 = Theta2(:,2:end)' * d_3 .* sigmoidGradient(z2(:,t));        %25*1
        
        %   ?(l):=?(l)+d(l+1)(a(l))T
        delta_3 = delta_3 + (d_3*a2(:,t)');                   %10 * 26
        delta_2 = delta_2 + (d_2*a1(:,t)');                   %25 * 401    , exclude bias unit in d_2 
                
    end;
    Theta1_grad = (1/m) * (delta_2);
    Theta2_grad = (1/m) * (delta_3);    

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    %step1: calculate regularization from 2:end - excluding bias unit
    
    reg1 = (lambda/m)*Theta1(:, 2:end);
    
    reg2 = (lambda/m)*Theta2(:, 2:end);
    
    %step2: add to theta_grad
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + reg1;
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + reg2;
    %Theta1_grad = (1/m) * (delta_2 + (lambda * Theta1));
    %Theta2_grad = (1/m) * (delta_3 + (lambda * Theta2));
    
    
% -------------------------------------------------------------
    %% 5. prepare sigmoid gradient function
    
    
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

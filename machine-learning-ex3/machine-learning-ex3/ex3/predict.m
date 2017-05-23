function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values - should be 5000
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%layer1
%add column 1's to the matrix
X = [ones(m,1), X]; %result dimension 5000 * 401

a1 = X';            %result dimension is 401 * 5000;

z2 = sigmoid(Theta1*a1);     %result dimension is 25 * 5000;


%layer2
%add column 1's to the matrix
a2 = [ones(1,m); z2]; %result dimension is 26 * 5000;

z3 = sigmoid(Theta2*a2);       %result dimension is 10 * 5000;

h = z3' ;              %result dimension is 5000 * 10;

[j, p] = max(h, [], 2); % find the max for each row.

% =========================================================================

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Calculate cost of X,y and theta plus reg
h = sigmoid(X*theta);
% exclude the first theta
theta1 = [0; theta(2:size(theta),:)];

% regulization = lambda/2m * sum of square theta_j
reg = (lambda/(2*m)) * sum(theta1.^2);

J = (1/m * sum((-y .* log(h)) - ((1-y) .*log(1-h)))) + reg;



%Calculate grad
  %loop through each feature of X
  for i = 1:size(X,2)
      %Calculate dJ(theta)/d(theta)
      reg_grad = (lambda/m) * theta1(i);
      grad(i) = ((1/m) * sum((h - y) .* X(:,i))) + reg_grad;
  end;
  


% =============================================================

end

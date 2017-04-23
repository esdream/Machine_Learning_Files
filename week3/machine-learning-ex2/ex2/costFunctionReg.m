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

% Calc the h_theta(x) and h_theta(x1)
h_x = sigmoid(X * theta);

% Calc the J
J = -1 / m * (y' * log(h_x) + (ones(m, 1) - y)' *  log(1 - h_x)) + lambda / (2 * m) * sum(theta(2:size(theta)) .^ 2);

% Calc the gradient
theta_except = zeros(size(theta), 1);
% not penalize theta(1), so set the tetha(1) to 0 before calculate the grad
theta_except(1) = theta(1);
grad = 1 / m * X' * (h_x - y) + lambda / m * (theta - theta_except);

% =============================================================

end

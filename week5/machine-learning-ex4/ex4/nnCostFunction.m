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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Transfor y(a m * 1 vector) to Y(a m * num_labels matrix)
length_y = length(y);
Y = zeros(m, num_labels);
for i = 1 : length_y
    Y(i, y(i)) = 1;
end

% Calc the A1
X = [ones(m, 1), X];
A1 = X;

% Calc the A2
A2 = sigmoid(A1 * Theta1');
A2 = [ones(size(A2, 1), 1), A2];

% Calc the H_x
H_x = sigmoid(A2 * Theta2');

% Calc the Cost Function of NN
for i = 1 : m
    J += - 1 / m * (Y(i, :) * log(H_x(i, :))' + (1 - Y(i, :)) * log(1 - H_x(i, :))');
end

% regularization of Cost Function
% Get the Square of each Theta, and do not penalize the bias unit
square_Theta1 = (Theta1 .^ 2)(:, 2:end);
square_Theta2 = (Theta2 .^ 2)(:, 2:end);
J = J + lambda / (2 * m) * (sum(square_Theta1(:)) + sum(square_Theta2(:)));


% Calc the Gradient of Theta
for i = 1 : m
    a_1 = A1(i, :)';
    a_2 = A2(i, :)';
    a_3 = H_x(i, :)';
    y = Y(i, :)';

    % Calc the delta_3
    delta_3 = a_3 - y;

    % Calc the delta_2
    delta_2 = Theta2' * delta_3 .* a_2 .* (1.0 - a_2);

    % do not penalize the delta(0)
    Theta2_grad += delta_3 * a_2';
    Theta1_grad += delta_2(2:end) * a_1';
end

Theta1_grad = 1 / m * Theta1_grad + lambda / m * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
Theta2_grad = 1 / m * Theta2_grad + lambda / m * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

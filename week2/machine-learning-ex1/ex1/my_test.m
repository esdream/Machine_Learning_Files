% test in development
data = load('ex1data2.txt');
X = data(:, 1:2);
fprintf('size of n: %d\n',size(X, 2));
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
disp(mu));
disp(sprintf('sigma: %d\n',sigma));

x_size = X(:,1);
x_numOfBedrooms = X(:,2);

fprintf('substract = %f\n', x_size - mean(x_size));
fprintf('std(x_size) = %f\n', std(x_size));
fprintf('norm = %f\n', (x_size - mean(x_size)) / std(x_size));
% numOfBedrooms_norm = x_numOfBedrooms .- mean(x_numOfBedrooms) ./ std(x_numOfBedrooms);

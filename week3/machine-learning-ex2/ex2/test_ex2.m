% load data
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

pos = find(y == 1); neg = find(y == 0);

disp('pos indices:');
disp(pos);
disp('neg indices:');
disp(neg);

clear
clc
demand=textread('hw2_max_demand.txt');
temp=textread('hw2_max_temp.txt');
a=length(demand);
index=round(0.7*a);
index_test=round(0.9*a);
normalize_temp = max(temp);
normalize_demand = max(demand);
demand = demand./normalize_demand;
temp= temp./normalize_temp;
lambda=0 : 0.1 : 5;
n = [2, 5:5:40];
for j = 1 : length(n)
for i = 1 : length(lambda)
theta = zeros(n(j),1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, temp(1:index), demand(1:index),lambda(i))), theta, options);
% test date
[J(i, j), ~] = costFunction(theta, temp(index+1:index_test), demand(index+1:index_test),0);
% performance
if i >= 2
if J(i, j)<J(i-1, j)
[J_per(j), ~] = costFunction(theta, temp(index_test+1:end), demand(index_test+1:end),lambda(i));
end
else
[J_per(j), ~] = costFunction(theta, temp(index_test+1:end), demand(index_test+1:end),lambda(i));
end
end
figure(1)
plot(lambda, J(:, j)); hold on;
xlabel('\lambda')
ylabel('Error')
end
n
J_per

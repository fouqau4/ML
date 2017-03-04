load('hw2_max_demand.txt');
load('hw2_max_temp.txt');

X_linear = [ ones( size( hw2_max_temp, 1 ), 1 ) hw2_max_temp ];
X_non_linear = [ X_linear ( hw2_max_temp .^ 2 ) ];
y = hw2_max_demand;

theta_linear = zeros( size( X_linear, 2 ), 1 );
theta_non_linear = zeros( size( X_non_linear, 2 ), 1 );

lambda = 10 ^ -2;
m = size( y, 1 );



J = ( y - X_linear * theta_linear  ) .^ 2 + lambda * theta_linear .^ 2;

max_iteration_time = 10 ^ 5;

for i = 1 : max_iteration_time
	grad_J = 2 * ( X_linear' * X_linear * theta_linear - X_linear' * y + lambda * theta_linear );
	theta_linear_new = theta_linear - alpha * grad_J;
	
end

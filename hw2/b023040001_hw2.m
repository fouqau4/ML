load('hw2_max_demand.txt');
load('hw2_max_temp.txt');

X_linear = [ ones( size( hw2_max_temp, 1 ), 1 ) hw2_max_temp ];
X_non_linear = [ X_linear ( hw2_max_temp .^ 2 ) ];
y = hw2_max_demand;

theta_linear = zeros( size( X_linear, 2 ), 1 );
theta_non_linear = zeros( size( X_non_linear, 2 ), 1 )

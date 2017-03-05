clc
clear

load('hw2_max_demand.txt');
load('hw2_max_temp.txt');

A = [ hw2_max_demand hw2_max_temp ];
figure( 'Name', 'temp/demand' );
scatter( hw2_max_temp, hw2_max_demand );
ylabel('demand');
xlabel('temperature');


X_linear = [ ones( size( hw2_max_temp, 1 ), 1 ) ( ( hw2_max_temp - mean( hw2_max_temp ) ) / range( hw2_max_temp ) ) ];
tempSqr2 = hw2_max_temp .^ 2;
X_non_linear = [ X_linear ( ( tempSqr2 - mean( tempSqr2 ) ) / range( tempSqr2 ) ) ];
y = hw2_max_demand;

alpha = 10 ^ -6;
lambda = 10 ^ -2;
epsilon = 10 ^ -3;
partition = size( y, 1 ) * [ 0.6 ; 0.3 ; 0.1 ]

X_linear_test = X_linear( 1 : partition(1), : );
X_non_linear_test = X_non_linear( 1: partition(1), : );
y_test = y( 1 : partition(1), : );

theta_linear = zeros( size( X_linear, 2 ), 1 );
theta_non_linear = zeros( size( X_non_linear, 2 ), 1 );


max_iteration_time = 10 ^ 5;

options = optimset( 'GradObj', 'on', 'MaxIter', max_iteration_time );
[ theta_linear, cost_linear ] = fminunc( @( t )( costFunction( t, X_linear_test, y_test, lambda ) ), theta_linear, options )
[ theta_non_linear, cost_non_linear ] = fminunc( @( t)( costFunction( t, X_non_linear_test, y_test, lambda ) ), theta_non_linear, options  )


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
y = [ hw2_max_demand ];

alpha = 10 ^ -6;
lambda = [10 ^ -2  10 ^ -3  10 ^ -4  10 ^ -5  10 ^ -6  10 ^ -7 ];
epsilon = 10 ^ -3;
ratio = [ 0.6 ; 0.3 ; 0.1 ]
partition = size( y, 1 ) * ratio

rangeBegin = 1;
rangeEnd = partition(1);

X_linear_test = X_linear( rangeBegin : rangeEnd, : );
X_non_linear_test = X_non_linear( rangeBegin : rangeEnd, : );
y_test = y( rangeBegin : rangeEnd, : );

rangeBegin = partition(1) + 1;
rangeEnd = partition(1) + partition(2);

X_linear_choose = X_linear( rangeBegin : rangeEnd, : );
X_non_linear_choose = X_non_linear( rangeBegin : rangeEnd, : );
y_choose = y( rangeBegin : rangeEnd, : );

rangeBegin = partition(1) + partition(2) + 1;

X_linear_perf = X_linear( rangeBegin : end, : );
X_non_linear_perf = X_non_linear( rangeBegin : end, : );
y_perf = y( rangeBegin : end, : ); 

theta_linear = zeros( size( X_linear, 2 ), size( lambda, 2 ) );
theta_non_linear = zeros( size( X_non_linear, 2 ), size( lambda, 2 ) );


max_iteration_time = 10 ^ 5;

options = optimset( 'GradObj', 'on', 'MaxIter', max_iteration_time );

for i = 1 : size( lambda, 2 )
	[ theta_linear( :, i ), cost_linear( i ) ] = fminunc( @( t )( costFunction( t, X_linear_test, y_test, lambda( i ) ) ), theta_linear( :, i ), options );
	[ theta_non_linear( :, i ), cost_non_linear( i ) ] = fminunc( @( t)( costFunction( t, X_non_linear_test, y_test, lambda( i ) ) ), theta_non_linear( :, i ), options  );
end

for i = 1 : size( lambda, 2 )
    J_linear(i) =   ( 1 / ( 2 * partition(2) ) ) * norm( X_linear_choose * theta_linear( :, i ) - y_choose ) .^ 2 + ( 1 / 2 ) * lambda(i) * norm( theta_linear( :, i ) ) .^ 2;

    J_non_linear(i) = ( 1 / ( 2 * partition(2) ) ) * norm( X_non_linear_choose * theta_non_linear( :, i ) - y_choose ) .^ 2 + ( 1 / 2 ) * lambda(i) * norm( theta_non_linear( :, i ) ) .^ 2;

end

fprintf("\n**********************************************************\n");
fprintf("\nlinear : \n");
lambda
theta_linear
cost_linear
J_linear
[ min_cost_linear, min_cost_linear_index ] = min(J_linear)
performance_linear = ( 1 / ( 2 * partition(3) ) ) * norm( X_linear_perf * theta_linear( :, min_cost_linear_index ) - y_perf ) .^ 2



fprintf("\n**********************************************************\n");
fprintf("\nnon-linear : \n");
lambda
theta_non_linear
cost_non_linear
J_non_linear
[ min_cost_non_linear, min_cost_non_linear_index ] = min(J_non_linear)
performance_non_linear = ( 1 / ( 2 * partition(3) ) ) * norm( X_non_linear_perf * theta_non_linear( :, min_cost_non_linear_index ) - y_perf ) .^ 2

figure(2)
plot(1:length(lambda), J_non_linear)


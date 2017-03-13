clc
clear

load('hw2_max_demand.txt');
load('hw2_max_temp.txt');



X_linear = [ ones( size( hw2_max_temp, 1 ), 1 ) ( ( hw2_max_temp - mean( hw2_max_temp ) ) / range( hw2_max_temp ) ) ];
tempSqr2 = hw2_max_temp .^ 2;
X_non_linear = [ X_linear ( ( tempSqr2 - mean( tempSqr2 ) ) / range( tempSqr2 ) ) ];
y = [ hw2_max_demand ];

alpha = 10 ^ -6;
lambda = [0:1:5 ];
epsilon = 10 ^ -3;
ratio = [ 0.6 ; 0.3 ; 0.1 ]
partition = size( y, 1 ) * ratio

rangeBegin = 1;
rangeEnd = partition(1);

X_linear_test = hw2_max_temp( rangeBegin : rangeEnd );
X_non_linear_test = hw2_max_temp( rangeBegin : rangeEnd, : );
y_test = y( rangeBegin : rangeEnd, : );

rangeBegin = partition(1) + 1;
rangeEnd = partition(1) + partition(2);

X_linear_choose = hw2_max_temp( rangeBegin : rangeEnd, : );
X_non_linear_choose = hw2_max_temp( rangeBegin : rangeEnd, : );
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
    [J_linear(i),~] = costFunction( theta_linear(:,i), X_linear_choose, y_choose, 0 );
	[J_non_linear(i),~]=costFunction(theta_non_linear(:,i), X_non_linear_choose,y_choose,0);

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


figure( 'Name', 'temp/demand' );
plot( hw2_max_temp, hw2_max_demand, 'x' );
ylabel('demand');
xlabel('temperature');

color = [ 'y' 'm' 'c' 'r' 'g' 'b' ]

figure( 'Name', 'J_linear');
hold on;
for i = 1 : length( lambda )
	bar( i, J_linear( i ), 0.3, color( i ) );
end
legend('lambda = 1e-2', 'lambda = 1e-3', 'lambda = 1e-4', 'lambda = 1e-5', 'lambda = 1e-6', 'lambda = 1e-7');
hold off;
ylabel( 'J(theta)' );
xlabel( 'lambda' );

figure( 'Name', 'J_non_linear' );
hold on;
for i = 1 : length( lambda )
	bar( i, J_non_linear( i ), color( i ), 0.3 );
end
legend('lambda = 1e-2', 'lambda = 1e-3', 'lambda = 1e-4', 'lambda = 1e-5', 'lambda = 1e-6', 'lambda = 1e-7');
hold off;
ylabel( 'J(theta)' );
xlabel( 'lambda' );

%pause

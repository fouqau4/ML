clc
clear

% part( a )-----------------------------------------------------------------------------------------

load('hw2_max_demand.txt');
load('hw2_max_temp.txt');

figure( 'Name', 'temp/demand' );
plot( hw2_max_temp, hw2_max_demand, 'x' );
ylabel('Peak Demand (GW)');
xlabel('High Temperature (F)');

% part( b )-----------------------------------------------------------------------------------------
temp = hw2_max_temp;
nmlz_temp =  ( hw2_max_temp - mean( hw2_max_temp) ) / range( hw2_max_temp );
demand = hw2_max_demand;
nmlz_demand = ( hw2_max_demand - mean( hw2_max_demand ) ) / range( hw2_max_demand );
m = size( demand, 1 );

temp_in_min = (min(temp)-mean(temp))/range(temp);
temp_in_max = (max(temp)-mean(temp))/range(temp);

alpha = 1e-1;
lambda = 0;
ratio = [ 0.7 0.2 0.1 ];
part = [ round( ratio( 1 ) * m ) round( sum( ratio( 1 : 2 ) ) * m ) round( sum( ratio( 1 : 3 ) ) * m )  ];

% linear

theta_linear = randn( 2, 1 );

grad = 1;

X_train = [ ones( part(1), 1 ) nmlz_temp( 1 : part(1) ) ];
y_train = nmlz_demand( 1 : part(1) );

while mean( abs( grad ) ) > 1e-5
	grad = ( 1 / part(1) ) * ( X_train' * ( X_train * theta_linear  - y_train ) ) + lambda .* [0; theta_linear(2:end)];
	theta_linear = theta_linear - alpha * grad;
end

theta_linear

figure(2)
plot(temp,demand,'*');
hold on
temp_in =[ temp_in_min - 1:0.1: 2 * temp_in_max]';
demand_est = [ones(length(temp_in),1),temp_in]*theta_linear;
plot(temp_in*range(temp)+mean(temp),demand_est*range(demand)+mean(demand) ,'r-');
axis([0, 100, 1.2, 3.2])

% non_linear

n = 10;

theta_non_linear = randn( n, 1 );

X_train = zeros( part(1), n );
for i = 1 : n
	X_train( :, i ) = nmlz_temp( 1 : part(1) ) .^ ( i - 1 );
end

grad = 1;

while mean( abs( grad ) ) > 1e-3
	grad = ( 1 / part(1) ) * ( X_train' * ( X_train * theta_non_linear - y_train ) ) + lambda .* [0;theta_non_linear(2:end)];
	theta_non_linear = theta_non_linear - alpha * grad;
end

figure(3)
plot(temp,demand,'*');
hold on
A = zeros( length(temp_in), n );
for i = 1 : n
	A(:,i) = temp_in .^ ( i - 1 );
end
demand_est = A*theta_non_linear;
plot(temp_in*range(temp)+mean(temp),demand_est*range(demand)+mean(demand) ,'r-');
axis([0, 100, 1.2, 3.2])


% part( d ) -----------------------------------------------------------------------------------------------------

n = 2;

theta_gd = zeros( n, 1 );
grad = ones( n, 1 );

X_train = nmlz_temp( 1 : part(1) );
y_train = nmlz_demand( 1 : part(1) );

alpha = 0.5;
lambda = 0.1

while mean( abs( grad ) ) > 1e-5
	[ J, grad ] = costFunction( theta_gd, X_train, y_train, lambda );
	theta_gd = theta_gd - alpha * grad;
end


max_iteration_time = 400;

theta_fmin = zeros( n, 1 );

options = optimset( 'GradObj', 'on', 'MaxIter', max_iteration_time );
[ theta_fmin, cost_fmin ] = fminunc( @( t )( costFunction( t, X_train, y_train, lambda ) ), theta_fmin, options );

[theta_gd theta_fmin ]


% part( e ) ----------------------------------------------------------------------------------------------------

lambda = [0:0.1:5];

X_train = nmlz_temp( 1 : part(1) );
y_train = nmlz_demand( 1 : part(1) );
X_test = nmlz_temp( part(1)+1 : part(2));
y_test = nmlz_demand( part(1)+1 : part(2) );

n = [2, 3:1:6];
color = ['r','b','k','y','m'];

for j = 1 : size( n, 2 )
	for i = 1 : size( lambda, 2 )
    	theta = zeros( n(j), size(lambda,2) );
	    [ theta(:,i), ~ ] = fminunc( @( t )( costFunction( t, X_train, y_train, lambda( i ) ) ), theta(:,i), options );
	   	[ J( i, j ), ~ ] = costFunction( theta(:,i), X_test, y_test, 0 );
	end
	
	figure(4)
	plot( lambda, J( :, j ), '-' );hold on;
	xlabel('\lambda')
	ylabel('error')
end
legend('linear', '3', '4', '5', '6' )

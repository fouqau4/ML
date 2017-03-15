clc; close all; clear;

D = load('ex3data2.txt');
D = D( randperm( size(D,1) ), : );


X = D( :, 1:2 ); y = D( :, 3 );
X = mapFeature( X( :, 1 ) , X( :, 2 ) );

theta_initial = zeros( size( X, 2 ), 1 );

m = size( y, 1 );

tra = round( 0.7 * m );
val = round( 0.9 * m );

lambda = [ 0:0.1:2 ];
caseNum = size( lambda, 2 );

options = optimset( 'GradObj', 'on', 'MaxIter', 400 );

theta = zeros( size( X, 2 ), caseNum );
J = zeros( 1, caseNum );
accuracy = zeros( 1, caseNum );
for i = 1 : caseNum
	% training
	[ theta( :, i ), ~ ] = fminunc( @(t)( costFunctionReg( t, X( 1:tra, : ), y( 1:tra, : ), lambda(i) ) ), theta_initial, options );
	% validation
	[ J(i), ~ ] = costFunctionReg( theta( :, i ), X( tra+1 : val, : ), y( tra+1 : val, : ), 0 );
	p = predict( theta( :, i ), X( tra+1 : end, : ) );
	accuracy(i) = mean( double( p == y( tra+1 : end, : ) ) ) * 100;
end

[ max_accuracy, ~ ] = max(accuracy)
max_index = find( accuracy == max_accuracy );

% testing
for i = 1 : size( max_index, 2 )
	p = predict( theta( :, max_index(i) ), X(val+1:end,:));
	per(i) = mean(double(p==y(val+1:end,:))) * 100;
end

% cost / lambda graph
figure();
hold on;
plot( lambda, J, 'LineWidth', 2 );
ylabel('cost');
xlabel('\lambda');
hold off;

% accuracy / lambda graph
figure();
hold on;
plot( lambda, accuracy, 'LineWidth', 2 );
plot( lambda(max_index), accuracy(max_index), 'ro', 'MarkerSize', 7, 'LineWidth', 3 );
ylabel('Accuracy ( validation )');
xlabel('\lambda');
hold off;

max_index
lambda(max_index)
per

figure();
hold on;
plot( lambda(max_index), per );
plot( lambda(max_index), per, 'rx', 'LineWidth', 3, 'MarkerSize', 7 );
ylabel('Accuracy ( performance )');
xlabel('\lambda');
hold off;

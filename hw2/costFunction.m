function [J, grad] = costFunction(theta, X, y, lambda )
m = length(y); 
n = length(theta);
grad = zeros( size( theta ) );


Phi = zeros( m, n );
for i = 1 : n
	Phi( :, i ) = X .^ ( i - 1 );
end

%J = ( 1 / ( 2 * m ) ) * norm( Phi * theta - y ) .^ 2 + ( 1 / 2 ) * lambda * norm( theta(2:end) ) .^ 2;
J = ( Phi*theta-y )'*(Phi*theta-y)/(2*m) + lambda/2*theta(2:end)'*theta(2:end); 
grad =( Phi' * ( Phi * theta - y ) )/m + lambda * [0;theta(2:end)];

end

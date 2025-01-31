function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%% Hint: A way to compute rho
[netIn, trExamples] =  size(data);

m = trExamples;

partCost = 0;


DW1 = zeros(size(W1));
DW2 = zeros(size(W2));
Db1 = zeros(size(b1));
Db2 = zeros(size(b2));

% rho accumulator vector
rhoAcc = zeros(size(W1,1),1);

%for i = 1:m
% y = x = a1 = 64x1
%    y = data(:,i);
%    x = y;
    
%    a1 = x;
% z2 = 25x64 * 64x1 + 25x1 = 25x1
%    z2 = W1*a1 + b1;
% a2 = 25x1
%    a2 = sigmoid(z2);
    
%    rhoAcc = rhoAcc + a2;
    
%end
% rho = 25x1
%rho = 1/m*rhoAcc;


% b1 = 25 x 1
% b2 = 64 x 1

% data = y = a1 : 64 x 10000
y = data;
a1 = data;
% z2 : 25x64 * 64x10000 .+ 25x1 = 25 x 10000
z2 = W1 * a1 .+ b1;
% a2 = 25 x 10000
a2 = sigmoid( z2 );
% z3 = 64x25 * 25x10000 .+ 64x1 = 64 x 10000
z3 = W2 * a2 .+ b2 ;
% a3 = 64 x 10000
a3 = sigmoid( z3 );
% hx = 64 x 10000
hx = a3;
% rho : 25 x 1
rho = sum( a2, 2 ) ./ m;

% delta3 : (64x10000 - 64x10000) .* 64x10000 .* 64x10000
delta3 = -( y - hx ) .* hx .* ( 1 .- hx );
% delta2 : ( (64x25)' * 64x10000 .+ (25x1 + 25x1) )
%           .* 25x10000 .* 25x10000  =  25 x 10000
delta2 = ( ( W2' * delta3 ) .+ ( beta .* ( -sparsityParam./rho + ( 1 - sparsityParam )./( 1 .- rho ) ) ) ) .* a2 .* ( 1 - a2 );

% D1 = 25x10000 * (64x10000)' = 25x64
D1 = delta2 * a1';
% D2 = 64x10000 * (25x10000)' = 64x25
D2 = delta3 * a2';

W1grad = 1 / m * D1 + lambda * W1;
W2grad = 1 / m * D2 + lambda * W2;

Db1 = delta2 * ones( m, 1 );
Db2 = delta3 * ones( m, 1 );

b1grad = 1 / m * Db1 + lambda * 0;
b2grad = 1 / m * Db2 + lambda * 0;

cost = 1 / m / 2 * norm( hx - y ) .^ 2 + lambda * ( sum((W1.^2)(:)) + sum((W2.^2)(:)) ) + ...
	   beta * compKL( sparsityParam, rho );

%%

%% Hint: A way to compute DW1, DW2, Db1, Db2



%for i = 1:m

%    y = data(:,i);
%    x = y;
% a1 = 64 x 1    
%    a1 = x;
% z2 = 25x64 * 64x1 + 25x1 = 25 x 1
%    z2 = W1*a1 + b1;
% a2 = 25 x 1
%    a2 = sigmoid(z2);
% z3 = 64x25 * 25x1 + 64x1 = 64 x 1
%    z3 = W2*a2 + b2;
% a3 = 64 x 1
%    a3 = sigmoid(z3);
%    hx = a3;
    
%    partCost = partCost + 1/2*norm(hx-y).^2;
% delta3 = 64 x 1    
%    delta3 = -(y-a3).*a3.*(1-a3);
% delta2 = (64x25)' * 64x1 = 25 x 1    
    % dalta 2 withour sparsity parameter is:
    % delta2 = W2'*delta3.*a2.*(1-a2);
%    delta2 = (W2'*delta3 + ...
%        beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho))).*a2.*(1-a2);
    
    %delta1 = W1'*delta2.*a1.*(1-a1);

% curW1grad = 100x1 * (64x1)' = 100 x 64
%    curW1grad = delta2*a1';
% curb1grad = 100 x 1
%    curb1grad = delta2;
%    curW2grad = delta3*a2';
%    curb2grad = delta3;
%    
%    DW1 = DW1 + curW1grad;
%    DW2 = DW2 + curW2grad;
%    Db1 = Db1 + curb1grad;
%    Db2 = Db2 + curb2grad;
%    
%end



% Compute cost:
% with lambda = 0 and beta = 0
% cost = 1/m * partCost;
% with beta = 0;
%cost = 1/m * partCost + lambda/2 * sum([sum(sum(W1.^2)) sum(sum(W2.^2))]);

% Add sparsity term
%cost = cost + beta*sum(compKL(sparsityParam,rho));

% Compute W1grad:
% with lambda = 0 and beta = 0
% W1grad = 1/m * DW1;
% with beta = 0;
%W1grad = 1/m * DW1 + lambda * W1;

% Compute W2grad:
% with lambda = 0 and beta = 0
% W2grad = 1/m * DW2;
% with beta = 0;
%W2grad = 1/m * DW2 + lambda * W2;

% Compute b1grad:
% with lambda = 0 and beta = 0
% b1grad = 1/m * Db1;
% with beta = 0;
%b1grad = 1/m * Db1;

% Compute b2grad:
% with lambda = 0 and beta = 0
% b2grad = 1/m * Db2;
% with beta = 0;
%b2grad = 1/m * Db2;
 

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function kl = compKL(sparsityParam, rho)

    kl = sum(sparsityParam.*log(sparsityParam./rho) +...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

end


%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

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

% size of the data
m = size(data,2);

% forward propagation
z2 = W1 * data + repmat(b1,1,m);
a2 = sigmoid(z2);
assert(size(a2,1) == hiddenSize);
assert(size(a2,2) == m);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = sigmoid(z3);
assert(size(a3,1) == visibleSize);
assert(size(a3,2) == m);

% average activation of the hidden layer
rho_hat = sum(a2, 2) / m;
assert(size(rho_hat(:),1) == size(a2,1));

% sparsity constraint vector
sparsity_delta = - sparsityParam ./ rho_hat + (1 - sparsityParam) ./ (1 - rho_hat);
assert(size(sparsity_delta, 1) == size(a2,1));

%sparsity penalty term
kl_sum = sum(sparsityParam * log(sparsityParam ./ rho_hat) + (1-sparsityParam)*log((1-sparsityParam)./(1-rho_hat)));


% backpropagation
% erro terms for the output layer (layer 3)
e3 = -(data - a3) .* a3 .* (1 - a3);
assert(size(e3,1)==size(a3,1));
assert(size(e3,2)==size(a3,2));

% erro terms for the hiddent layer (layer 2) with sparsity_delta
e2 = (W2' * e3 + repmat(beta * sparsity_delta,1,m)).* a2 .* (1 - a2);
assert(size(e2,1)==size(a2,1));
assert(size(e2,2)==size(a2,2));

% compute the W1grad and b1grad
W1grad = e2 * data' / m + lambda * W1;
b1grad = sum(e2,2) / m;


% compute the W2grad and b2grad
W2grad = e3 * a2' / m + lambda * W2;
b2grad = sum(e3,2) / m;

% fprintf('assert the sizes of the gradients\n');
% assert the sizes of the gradients
assert(size(W1grad(:),1) == hiddenSize * visibleSize);
assert(size(W2grad(:),1) == hiddenSize * visibleSize);
assert(size(b1grad(:),1) == hiddenSize);
assert(size(b2grad(:),1) == visibleSize);

% sum of squared error term, weight decay term and sparsity penalty term
temp = (data - a3).^2;
cost = sum(temp(:)) / 2;
cost = cost / m + lambda / 2 * (sum(W1(:).^2) + sum(W2(:).^2)) + beta * kl_sum;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];



end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end


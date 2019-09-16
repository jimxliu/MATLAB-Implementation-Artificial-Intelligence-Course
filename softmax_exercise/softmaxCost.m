function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% exp of each class of each sample
M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
z = exp(M);
assert(size(z(:),1) == numClasses * numCases);

% softmax normalization factor
normfac = sum(z,1);
assert(size(normfac,1) == 1);
assert(size(normfac,2) == numCases);

% hypothesis function
h = z ./ normfac;
assert(size(h(:),1) == numClasses * numCases);

% cost function
temp = groundTruth .* log(h); % all zeros except the correct class for each sample
assert(size(temp(:),1) == numClasses * numCases);
cost = -1 / size(data,2) * sum(temp(:)) + lambda / 2 * sum(theta(:).^2);

% gradient
thetagrad = -((groundTruth - h) * data') / size(data,2) + lambda * theta;
assert(size(thetagrad,1) == numClasses);
assert(size(thetagrad,2) == inputSize);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


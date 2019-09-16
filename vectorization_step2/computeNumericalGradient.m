function numgrad = computeNumericalGradient(J, theta, origrad)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
%  n = 100;
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
% numgrad = zeros(n,1);
% position = zeros(n,1);
%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
EPSILON = 0.0001;

fprintf('computing the nmuerical gradient...\n');
fprintf('randomly choose 100 weights to check...\n');
for i = 1:size(theta,1)
%     position(x) = i;
%     fprintf('%dth random weight \n',i);
    theta(i) = theta(i) + EPSILON;
    [cost_plus, grad_plus] = J(theta);
    theta(i) = theta(i) - 2 * EPSILON;
    [cost_minus, grad_minus] = J(theta);
    theta(i) = theta(i) + EPSILON;
    numgrad(i) = (cost_plus - cost_minus) / (2*EPSILON);

    fprintf('numgrad(%d): %f, origrad(%d): %f\n', i, numgrad(i), i, origrad(i));

end
fprintf('Done\n');
%% ---------------------------------------------------------------
end

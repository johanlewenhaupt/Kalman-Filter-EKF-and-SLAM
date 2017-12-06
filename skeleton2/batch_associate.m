% function [c,outlier, nu_bar, H_bar] = batch_associate(mu_bar,sigma_bar,z,M,Lambda_m,Q)
% This function should perform the maximum likelihood association and outlier detection.
% Note that the bearing error lies in the interval [-pi,pi)
%           mu_bar(t)           3X1
%           sigma_bar(t)        3X3
%           Q                   2X2
%           z(t)                2Xn
%           M                   2XN
%           Lambda_m            1X1
% Outputs: 
%           c(t)                1Xn
%           outlier             1Xn
%           nu_bar(t)           2nX1
%           H_bar(t)            2nX3
function [c,outlier, nu_bar, H_bar] = batch_associate(mu_bar,sigma_bar,z,M,Lambda_m,Q)
n = length(z); N = length(M);
c = zeros(1,n); outlier = zeros(1,n); nu_bar = zeros(n,1);
H_bar = zeros(n,3); H = zeros(2,3,N); S = zeros(2,2,N);
nu = zeros(2,N); phi = zeros(1,N); z_hat = zeros(2,N);
D = zeros(1,N);
for i = 1:1:n
    for j = 1:1:N
        z_hat(:,j) = observation_model(mu_bar,M,j);
        H(:,:,j) = jacobian_observation_model(mu_bar, M, j, z_hat, 1);
        S(:,:,j) = H(:,:,j)*sigma_bar*H(:,:,j)'+Q;
        nu(:,j) =z(:,i)-z_hat(:,j); 
        nu(2,j) = mod(nu(2,j) +  pi,2 * pi) - pi;
        D(j) = nu(:,j)'*inv(S(:,:,j))*nu(:,j);
        phi(j) = (det(2*pi*S(:,:,j)))^(-0.5)*exp(-0.5*D(j));
        
    end
    [ma,a]=max(phi); c(i) = a;
    outlier(i) = (D(c(i)) >= Lambda_m);
    nu_bar(2*i-1) = nu(1,c(i));
    nu_bar(2*i) = nu(2,c(i));
    H_bar(2*i-1,:) = H(1,:,c(i));
    H_bar(2*i,:) = H(2,:,c(i));
end


end
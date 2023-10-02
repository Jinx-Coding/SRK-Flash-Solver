
% ------------------- SRK Mixture Flash equilibrium -------------------- %

% Correra Giovanni - 2023 %

clc
close all
clear variables

OPTIONS = optimset('Display','off','MaxIter',1e20,'MaxFunEvals',1e20...
    ,'TolFun',1e-10,'Algorithm','levenberg-marquardt');

% ------------------------------ Data ---------------------------------- %

precision = 1e-20;

zF = [0.013148, 0.020329, 0.00079, 0.965733] ; % Feed molar fractions %
F = 2559.96; % Total Molar flowrate in (kmol/d)
F = zF * F; % Molar flowrate for single compound (kmol/d)
P = 1.01325; % System pressure in (bar)
T = 308.15; % System temperature in (K)

Tc = [190.4, 304.1, 373.2, 647.3]; % Critical temperatures in (K)
Pc = [46,73.8,89.4,221.2]; % Critical pressures in (bar)
om = [0.011,0.239,0.081,0.344]; % Omega values (-)

% -------------------------- Data check -------------------------------- %

if sum(zF)~=1
    fprintf('ERROR : feed molar fractions do not add up to 1\n')
    return
end

% -------------------------- First guess ------------------------------- %

% Detecting lowest volatile component from critical temperature %

[~,index] = max(Tc);

% Assuming everything is vapour except lowest volatile compound %

alpha_fg = (sum(F)-F(index)) / sum(F);

V_fg = alpha_fg * sum(F);

% Liquid fractions vector assuming all lowest volatile compound as liquid %

x_fg = zeros(1,length(zF));
x_fg(index) = 1;

% Vapour fractions vector assuming all volatile components as vapour %

y_fg = F/V_fg;
y_fg(index) = 0;

% -------------------------- Main script ------------------------------- %

error = 1;
j = 0;

while error > precision
    
    j = j+1;

    [ZV,AS,BS,AV,BV] = SRK(T,P,Tc,Pc,om,y_fg,1);
    [ZL,~,~,AL,BL] = SRK(T,P,Tc,Pc,om,x_fg,2);

    phiV = fugacity(ZV,AS,BS,AV,BV);
    phiL = fugacity(ZL,AS,BS,AL,BL);

    K = phiL./phiV;

    alpha = fsolve(@(alpha) ratchford_rice(alpha,zF,K),alpha_fg,OPTIONS);
    x = zF./(1+alpha.*(K-1));
    y = K.*zF./(1+alpha.*(K-1));

    error = abs(alpha-alpha_fg);

    alpha_fg = alpha;
    x_fg = x;
    y_fg = y;

    control1 = sum(y);
    control2 = sum(x);

end

% ------------------------ Post - Processing --------------------------- %

fprintf('N iter = %.0f\n',j)
fprintf('\n')

fprintf('alpha = %.12f\n',alpha)
fprintf('\n')

fprintf('x, CH4 = %.12f',x(1))
fprintf('   y, CH4 = %.12f\n',y(1))
fprintf('x, CO2 = %.12f',x(2))
fprintf('   y, CO2 = %.12f\n',y(2))
fprintf('x, H2S = %.12f',x(3))
fprintf('   y, H2S = %.12f\n',y(3))
fprintf('x, H2O = %.12f',x(4))
fprintf('   y, H2O = %.12f\n',y(4))
fprintf('sum(x) = %.12f',control2)
fprintf('   sum(y) = %.12f\n',control1)

% --------------------------- Functions -------------------------------- %

function [Z,AS,BS,A,B] = SRK(T,P,Tc,Pc,om,y,phase)

% SRK solving function with basic mixture laws %
% a(i,j) = (a(i)*a(j))^0.5 %
% b(i,j) = (b(i)+b(j))/2 %

% Phase = 1 (vapour), Phase = 2 (liquid) %

R = 8.3145;
RT = R*T;
RTc = R*Tc;

S = 0.48 + 1.574*om - 0.176*om.^2;
k = (1 + S.*(1-sqrt(T./Tc))).^2;
a = (0.42748*k.*RTc.^2)./Pc;
b = 0.08664*RTc./Pc;
AS = a.*P/(RT)^2;
BS = b.*P/(RT);
aM = sqrt(a'*a);
bM = zeros(length(Tc),length(Tc));
for i = 1:length(Tc)
    for j = 1:length(Tc)
        bM(i,j) = (b(i)+b(j))/2;
    end
end
am = y*aM*y';
bm = y*bM*y';
A = am*P/(RT)^2;
B = bm*P/(RT);
alfa = -1;
beta = A-B-B^2;
gamma = -A*B;

% Analytic solution %

p = beta - (alfa^2)/3;
q = 2*(alfa^3)/27 - alfa*beta/3 + gamma;
q2 = q/2;
a3 = alfa/3;
D = (q^2) / 4 + p^3 / 27;

if D>0
    Z1 = nthroot((-q2+sqrt(D)),3) + nthroot((-q2-sqrt(D)),3) - a3;
    Z = [Z1,Z1,Z1];
elseif D == 0
    Z1 = -2*nthroot(q2,3) - a3;
    Z2 = nthroot(q2,3) - a3;
    Z = [Z1,Z2,Z2];
elseif D<0
    r = sqrt(-p^3 / 27);
    teta = acos(-q2*sqrt(-27/p^3));
    Z1 = 2*nthroot(r,3)*cos(teta/3)-a3;
    Z2 = 2*nthroot(r,3)*cos((2*pi+teta)/3)-a3;
    Z3 = 2*nthroot(r,3)*cos((4*pi+teta)/3)-a3;
    Z = [Z1,Z2,Z3];
end

Z = sort(Z);

if phase == 1
    Z = max(Z);
elseif phase == 2
    Z = min(Z);
end

end

function phi = fugacity(Z,AS,BS,A,B)

% Takes SRK parameters (both single component and mixture) and determines
% vapour and liquid phases fugacities %

lnphi = (Z-1).*BS./B   ...
        + (A./B).*((BS./B)-2.*(AS./A).^0.5) ...
        .*log((Z+B.*(1+2^0.5))./(Z+B.*(1-2^0.5))) ...
        -log(Z-B);

phi = exp(lnphi);

end

function fun = ratchford_rice(alpha,zF,K)

% Ratchford Rice function %

single = zF.*(K-1)./(1+alpha.*(K-1));
fun = sum(single);

end


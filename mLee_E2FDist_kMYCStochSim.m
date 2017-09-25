%%This code contains the Langevin equations (SDEs) for the model

function x = mLee_E2FDist_kMYCStochSim(dt, Tspan, x0, Sfinal, kMYC, paraset, sigma, delta)

% Initializing constants
kM = kMYC;
kE = paraset(2);
kCD = paraset(3);
kCDS = paraset(4);
kR = paraset(5);
kb = paraset(6);
kCE = paraset(7);
KS = paraset(8);
kRE = paraset(9);

dM = paraset(10);
dE = paraset(11);
dCD = paraset(12);
dCE = paraset(13);
dR = paraset(14);
dRP = paraset(15);
dRE = paraset(16);

kP1 = paraset(17);
kP2 = paraset(18);
kP3 = paraset(19);
kP4 = paraset(20);
kDP = paraset(21);

KR = paraset(22);
KM = paraset(23);
KE = paraset(24);
KCD = paraset(25);
KCE = paraset(26);
KRP = paraset(27);

% Initializing answer matrix
x = zeros(length(Tspan),7);
x(1,:) = x0;

for i=1:length(Tspan)-1
    % Creating variables for the concentrations
    M=x(i,1); E=x(i,2); CD=x(i,3); CE=x(i,4); R=x(i,5); RP=x(i,6); RE=x(i,7);
    
    % Creating a vector of intrinsic noise values for the 6 phosphorylation/dephosphorylation terms
    randnum = randn(1,6);
    
    % Creating placeholders for common terms in the SDEs
    s = Sfinal/(KS+Sfinal);
    myc = M/(KM+M);
    e2f = E/(KE+E);
    rep = KR/(KR+M);
    r1 = (kP1*CD*R)/(KCD+R);
    r2 = (kP2*CE*R)/(KCE+R);
    r3 = (kP3*CD*RE)/(KCD+RE);
    r4 = (kP4*CE*RE)/(KCE+RE);
    r5 = kRE*R*E;
    r6 = (kDP*RP)/(KRP+RP);
    
    % Naive solution for the SDEs
     x(i+1,1) = M + dt*(kM*s - dM*M) +...
         sigma * (sqrt(dt*kM*s)*randn - sqrt(dt*dM*M)*randn) +...
         delta * randn * dt;
     x(i+1,2) = E + dt*((kE*myc*e2f + kb*myc)*rep + r3 + r4 - dE*E - r5)... 
         + sigma * (sqrt(dt*(kE*myc*e2f + kb*myc)*rep)*randn + ...
         sqrt(dt*r3)*randnum(3) + sqrt(dt*r4)*randnum(4)...
         - sqrt(dt*dE*E)*randn - sqrt(dt*r5)*randnum(5)) +...
         delta * randn * dt;
     x(i+1,3) = CD + dt*(kCD*myc + kCDS*s - dCD*CD) +...
         sigma * (sqrt(dt*kCD*myc)*randn + sqrt(dt*kCDS*s)*randn - ...
         sqrt(dt*dCD*CD)*randn) +...
         delta * randn * dt;
     x(i+1,4) = CE + dt*(kCE*e2f - dCE*CE) +...
         sigma * (sqrt(dt*kCE*e2f)*randn - sqrt(dt*dCE*CE)*randn) +...
         delta * randn * dt;
     x(i+1,5) = R + dt*(kR + r6 - r5 - r1 - r2 - dR*R) +...
         sigma * (sqrt(dt*kR)*randn + sqrt(dt*r6)*randnum(6) - ...
         sqrt(dt*r5)*randnum(5) - sqrt(dt*r1)*randnum(1) - ...
         sqrt(dt*r2)*randnum(2) - sqrt(dt*dR*R)*randn) +...
         delta * randn * dt;
     x(i+1,6) = RP + dt*(r1 + r2 + r3 + r4 - r6 - dRP*RP) +...
         sigma * (sqrt(dt*r1)*randnum(1) + sqrt(dt*r2)*randnum(2) + ...
         sqrt(dt*r3)*randnum(3) + sqrt(dt*r4)*randnum(4) - ...
         sqrt(dt*r6)*randnum(6) - sqrt(dt*dRP*RP)*randn) +...
         delta * randn * dt;
     x(i+1,7) = RE + dt*(r5 - r3 - r4 - dRE*RE) +...
         sigma * (sqrt(dt*r5)*randnum(5) - sqrt(dt*r3)*randnum(3) - ...
         sqrt(dt*r4)*randnum(4) - sqrt(dRE*RE)*randn) +...
         delta * randn * dt;
     x(i+1, find(x(i+1,:)<0))=0; %#ok<FNDSB>
 end
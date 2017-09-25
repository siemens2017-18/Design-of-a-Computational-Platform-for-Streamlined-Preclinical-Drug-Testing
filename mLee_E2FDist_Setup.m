rng('shuffle');

kM=1;        % uM/hr,           MYC synthesis rate (serum)
kE=0.15;     % uM/hr,    free,  E2F synthesis rate (MYC and autoregulation)
kCD=0.2;     % uM/hr,    free,  CycD synthesis rate (driven by MYC)
kCDS=0.2;    % uM/hr,    free,  CycD synthesis rate (driven by serum)
kR=0.35;     % uM/hr,    free,  Constitutive Rb synthesis rate
kb=0.001;    % uM/hr,    free,  E2F synthesis rate (MYC only)
kCE=0.5;     % uM/hr,    free,  CycE synthesis rate (driven by E2F)
KS=0.3;      % uM        free,  Half-maximal serum concentration
kRE=30;      % /(uM*hr), free,  RB-E2F synthesis rate

dM=0.7;      % /hr,                MYC decay rate
dE=0.25;     % /hr,                E2F decay rate
dCD=1.5;     % /hr,                CycD decay rate
dCE=1.5;     % /hr,                CycE decay rate
dR=0.06;     % /hr,                Rb decay rate
dRP=0.06;    % /hr, assumed = dR,  Phosphorylated Rb decay rate
dRE=0.03;    % /hr, assumed > Rb,  Rb-E2F complex decay rate

kP1=18;      % /hr,   Rb phosphorylation rate (CycD)
kP2=18;      % /hr,   Rb phosphorylation rate (CycE)
kP3=18;      % /hr,   E2F dissociation from Rb-E2F complex rate (CycD phosphorylation)
kP4=18;      % /hr,   E2F dissociation from Rb-E2F complex rate (CycE phosphorylation)
kDP=3.6;     % uM/hr, Dephosphorylation rate

KR=100;      % uM, experimental,     EFm half-maximal repression by MYC
KM=0.15;     % uM, Myc/Max estimate, MYC half-maximal concentration
KE=0.15;     % uM, assumed=KM,       E2F half-maximal concentration
KCD=0.92;    % uM,                   CycD half-maximal concentration
KCE=0.92;    % uM, assumed=KCD,      CycE half-maximal concentration
KRP=0.01;    % uM, typical value,    Phosphorylated Rb half-maximal concentration

U=10^-14; N=6.02e17;
Z=U*N;

kM=kM*Z; kE=kE*Z; kCD=kCD*Z; kCDS=kCDS*Z; kR=kR*Z; kb=kb*Z; kCE=kCE*Z; KS=KS*Z; kRE=kRE/Z;
kDP=kDP*Z;
KR=KR*Z; KM=KM*Z; KE=KE*Z; KCD=KCD*Z; KCE=KCE*Z; KRP=KRP*Z;

Sfinal=Sfinal*Z;

paraset=[kM, kE, kCD, kCDS, kR, kb, kCE, KS, kRE, ...
    dM, dE, dCD, dCE, dR, dRP, dRE, ...
    kP1, kP2, kP3, kP4, kDP, ...
    KR, KM, KE, KCD, KCE, KRP];

%  Init[M, E, CD,CE,R,   RP,RE  ]
x0 = Z*[0, 0, 0, 0, 0.4, 0, 0.25];
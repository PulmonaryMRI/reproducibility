function gate = ute_k0_gate_extract_pr_v2(k0, ref)
%% Use K0 to extract self-gating signal
% ref is a reference signal (normalized bellows)

%find the largest variation coils
%tmp = std(abs(k0),1);  
%ind = tmp == max(tmp);
figure, plot(abs(k0))

a = abs(fftshift(fft(abs(k0),[],1),1));
sz = size(a);
N = sz(1);
b = ((-1/2):1/N:(1/2-1/N)) * 1/(2.4e-3);
b=b';

% max freq at heart rate range
a1 = a;
a1(b<0.7 | b > 2,:)=NaN;
[~,i1] = max(a1,[],1,'omitnan');
ind1 = mode(i1);
% max freq at breathing rate
a2 = a;
a2(b>0.5 | b < 0.03,:)=NaN;
[~,i2] = max(a2,[],1,'omitnan');
ind2 = mode(i2);

%ratio = a(ind2,:) ./ a(ind1,:);
for i = 1:sz(2)
    ratio(i) = a(i2(i),i) / a(i1(i),i);
end

[~,ind] = max(ratio .* (i1> (ind1-10)) .* (i1 < (ind1+10)).* (i2>(ind2-10)).*(i2<(ind2+10)) );
%figure, plot(b',a)

% use one coil
%hh = fir1(512,1/250);
%hh = fir1(1024,1/5000);
rawdata = k0(:,ind); % 2nd coil has the strongest respiratory signal in an 8-channel chest coil
%tmp = [rawdata(400:-1:1);rawdata;rawdata(end:-1:end-255)];
tmp = [rawdata(5000:-1:1);rawdata;rawdata(end:-1:end-511)];

% a = abs(fftshift(fft(abs(rawdata))));
%N = length(rawdata);
%b = ((-1/2):1/N:(1/2-1/N)) * 1/(2.4e-3);
%figure, plot(b',a)

[yupper,ylower] = envelope(abs(tmp),500,'peak');
dk0 = (yupper+ylower) / 2;
%gate = dk0((400+257):(400+256+length(rawdata)));
gate = dk0(5000 + (1:length(rawdata)));
% normalize
gate = (gate-mean(gate))/std(gate);

if(sum(gate.*ref)< 0) 
    gate = -gate;
end

%% Adjust drifts according using Asymmetric Least-Squares
%bline = baseline(gate,10^9,0.01);
%gate = gate - bline;
% normalize
%gate = (gate-mean(gate))/std(gate);

end

function z = baseline(y ,lambda, p)
% applies asymmetric least squares to estimate baseline, z.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this code is taken from Eilers, Boelens 'Baseline Correction with 
% Asymmetric Least Squares Smoothing' 2005
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% .001 < p < .1
% 10e2 < lambda < 10^9

% for UTE self gating: (10^9, 0.01 )
m = length(y);
D = diff(speye(m), 2);
w = ones(m,1);
maxIter = 5;
 
for it = 1:maxIter
    W = spdiags(w, 0, m, m);
    C = chol(W + lambda * (D' * D));
    z = C \ (C' \ (w .* y)); %C \ (w .* y);
     
    % perform estimation without inverting C?
    %z = cgs(W+lambda*D'*D,w.*y,[],100);
    
    w = p * (y > z) + (1 - p) * (y < z);
end
 
end

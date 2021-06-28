function Yout = enhancment(Y,enhance_strengh, alpha, block_size)


[ ver hor ] = size(Y);

Yout = Y;

Rvd = 6;
Pic = 4;

theta = 2*atan(1/2/Pic/Rvd);
s_pi_v = ones(block_size,1)*sqrt(2/block_size);
s_pi_v(1) = sqrt(1/block_size);
s_pi_h = s_pi_v';

s = 0.25;
r = 0.6;
a=2.6; b=0.0192; c=0.114; d=1.1;

Lmax = 256;
Lmin = 0;
G = 256;
for v = 1:block_size
    for h = 1:block_size
        wij(v,h) = 1/(2*sqrt(2)*block_size*theta) * sqrt( v*v + h*h );
        wi0 = 1/(2*sqrt(2)*block_size*theta) * sqrt( 0 + h*h );
        w0j = 1/(2*sqrt(2)*block_size*theta) * sqrt( v*v + 0 );
        angle(v,h) = abs( asin( 2*wi0*w0j/wij(v,h)/wij(v,h) ) );
        A_f = a*(b+c*wij(v,h))*exp(-(c*wij(v,h))^d);        
        T_base(v,h) = s/A_f*G/(Lmax-Lmin)/s_pi_v(h)/s_pi_h(v)/(r+(1-r)*cos(angle(v,h))^2);        
    end
end
T_base2 = T_base.^2;

count = 0;
for num = 1:2*block_size^2
    if sqrt(num) <= block_size
        for vv = 1:sqrt(num)
            hh = sqrt(num+1-vv^2);
            if (floor(hh)-hh) == 0
                count = count+1;
                band_1d(count) = num;
                v_order(count) = vv;
                h_order(count) = hh;                
            end
        end
    else
        for vv = floor(sqrt(num+1-block_size^2)):block_size
            hh = sqrt(num+1-vv^2);
            if (floor(hh)-hh) == 0
                count = count+1;
                band_1d(count) = num;
                v_order(count) = vv;
                h_order(count) = hh;
            end
        end
    end
end

count = 0;
min_num = 0;
for num = 1:block_size^2
    if band_1d(num) == min_num
        band_order(num) = count;
    elseif band_1d(num) > min_num
        count = count + 1;
        min_num = band_1d(num);
        band_order(num) = count;
    end
end
band_order_max = max(band_order);


[ ver hor ] = size(Y);


tmp_dif = zeros(ver, hor);
tmp_rat = zeros(ver, hor);
tmp_anl = zeros(ver, hor);
ER_map = zeros(ver, hor);
H_map = zeros(ver, hor);
V_map = zeros(ver, hor);
mH_map = zeros(ver, hor);
mV_map = zeros(ver, hor);
TexE_map = zeros(ver, hor);

blambda = [0 0.0625 0.1250 0.1875 0.2500 0.3125 0.3750 0.4375 0.5000 0.5625 0.6250 0.6875 0.7500 0.8125 0.8750 0.9375];
% blambda = [0 0.1250 0.2500 0.3750 0.5000 0.6250 0.7500 0.8750 0.9375 0.8125 0.6875 0.5625 0.4375 0.3125 0.1875  0.0625];

%% Enhancement
for v=1:block_size:ver-block_size+1
    for h=1:block_size:hor-block_size+1

        % original block processing
        blockDCT = dct2(Y(v:v+block_size-1,h:h+block_size-1));
        blockDCTabs = abs(blockDCT);
        blockDCT_sq = blockDCT.^2;

        Ti = ( sum(sum(blockDCTabs)) - blockDCTabs(1,1)+0.0001);
        Htmp = sum( blockDCTabs(2:end,1) ) / Ti +0.0001;
        Vtmp = sum( blockDCTabs(1,2:end) ) / Ti +0.0001;
        
        H_map(v:v+block_size-1,h:h+block_size-1) = Htmp / (Htmp + Vtmp);
        V_map(v:v+block_size-1,h:h+block_size-1) = Vtmp / (Htmp + Vtmp);

        mH_map(v:v+block_size-1,h:h+block_size-1) = sum( blockDCTabs(2:end,1) );
        mV_map(v:v+block_size-1,h:h+block_size-1) = sum( blockDCTabs(1,2:end) );
        
        Grad = Htmp+Vtmp;

        % % % Enhancement % % %
        blockDCT_enh = blockDCT;
        R = zeros(block_size,block_size);

        band_energy=zeros(band_order_max,1);
        band_energy_enh=zeros(band_order_max,1);

        band_energy(1) = blockDCT_sq(1,1);
        band_energy_enh(1) = band_energy(1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        st_band = floor(block_size/3)+1; 
        
        dlambda = ones(block_size,block_size)*enhance_strengh; %*min(2,(TexE_map(v,h)./(TexE_map(v,h)+300)*300/(ER+300)+1));
%         dlambda = repmat(blambda, block_size,1)*enhance_strengh+1; %*min(2,(TexE_map(v,h)./(TexE_map(v,h)+100)*100/(ER+100)+1)) + 1;
        for bb = 1:st_band
            for cc = 1:st_band-bb
                dlambda(cc,bb) = 1.0;
            end
        end
        lambda_alpha = power( ( abs(blockDCT)+0.0001 ) / blockDCT(1,1), alpha-1); % alpha looting

        OriEnergyVer = sum(sum((blockDCT(:,1:1)).^2)) - blockDCT(1,1)^2;%sum(sum((blockDCT(:,1:1)).^2));
        EnEnergyVer = OriEnergyVer;
        dctVerY = blockDCT;       

        lamdaVer= dlambda;
        RVer=ones(block_size,block_size);
        for vv = 2:block_size  % u is index
            RVer(vv,:) = ((sum(T_base2(vv,:)) + EnEnergyVer + sum( sum( lambda_alpha(vv,:).*blockDCT(vv,:).*lambda_alpha(vv,:).*blockDCT(vv,:) ) ) ) ...
                        /(sum(T_base2(vv,:)) + OriEnergyVer + sum( sum( blockDCT(vv,:).*blockDCT(vv,:) ) ) ))^(1/2.4);
                    
            dctVerY(vv,:) = lamdaVer(vv,:).*RVer(vv,:).*blockDCT(vv,:);    
            OriEnergyVer = OriEnergyVer + sum( sum( blockDCT(vv,:).*blockDCT(vv,:) ) );
            EnEnergyVer = EnEnergyVer + sum( sum( dctVerY(vv,:).*dctVerY(vv,:) ) );
        end
        
        OriEnergyHor = sum(sum((blockDCT(1:1,:)).^2)) - blockDCT(1,1)^2;%sum(sum((blockDCT(1:1,:)).^2));
        EnEnergyHor = OriEnergyHor;
        dctHorY = blockDCT;
        lamdaHor = dlambda';
        RHor=ones(block_size,block_size);
        for hh = 2:block_size % v is index 
            RHor(:,hh) = ((sum(T_base2(:,hh)) + EnEnergyHor + sum( sum( blockDCT(:, hh).*lambda_alpha(:,hh).*blockDCT(:, hh) ) ) ) ...
                        /(sum(T_base2(:,hh)) + OriEnergyHor + sum( sum( blockDCT(:, hh).*blockDCT(:, hh) ) ) ))^(1/2.4);
                    
            dctHorY(:, hh) = lamdaHor(:,hh).*RHor(:,hh).*blockDCT(:,hh);
            OriEnergyHor = OriEnergyHor + sum( sum( blockDCT(:, hh).*blockDCT(:, hh) ) );
            EnEnergyHor = EnEnergyHor + sum( sum( dctHorY(:, hh).*dctHorY(:, hh) ) );
        end        
        blockDCT_enh =   Htmp / (Htmp + Vtmp)*dctHorY + Vtmp / (Htmp + Vtmp)*dctVerY;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tmp_anl(v:v+block_size-1,h:h+block_size-1) = blockDCT_enh./(blockDCT+0.000001);
        
        blockDCTtmp = blockDCT;
        blockDCTtmp(1,1) = 0;
        
        max_enh = max(max(abs(blockDCT_enh - blockDCT)));
        dif_enh = abs(blockDCT_enh - blockDCT);
        
        tmp_dif(v:v+block_size-1,h:h+block_size-1) = abs(blockDCTtmp);

        Yout(v:v+block_size-1,h:h+block_size-1) = max(0,min(255,idct2(blockDCT_enh)));

    end
end



end
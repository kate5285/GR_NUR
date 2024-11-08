clear all;close all;
%%
% file_path = "C:\Users\kate5\Downloads\without weird data Growth rate and nutrient uptake (Scrippsiella acuminata) 2022 0913 OJH.xlsx";
file_path ="C:\Users\kate5\Downloads\without weird data Growth rate and nutrient uptake (Prorocentrum triestinum) 2022 0913 OJH.xlsx";
% speciesname="S. acuminata";
speciesname="P. triestinum";
P = xlsread(file_path, 4, 'W2:W61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기
N= xlsread(file_path, 4, 'K2:K61');
conc= xlsread(file_path, 4, 'E2:E61');

%튀는 값 제외--그래도 여전히 아웃라이어가 생김-> 왜지??
%for s.a.
eyy=0;
if speciesname=="S. acuminata"
startdate=4;
else
startdate=1;
end% 첫 두 날들을 빼서 3일부터 시작,즉 이 변수 값은 4임. acclimation.p.t.의 경우 다 보니까 1이라고 치면 됨
%% figure 2B; for po4 and gr
bottlePnut=cell(3,1); % 그 날 마다의 P양
for k=1:3
    bottlenuts=[];
for i=1:(numel(P)+1)/4
  bottlenuts=[bottlenuts;P(4*(i-1)+k,1)];
end
  bottlePnut{k}=bottlenuts;
end

bottleconc=cell(3,1);
for k=1:3
    bottleconcs=[];
for i=1:(numel(conc)+1)/4
  bottleconcs=[bottleconcs;conc(4*(i-1)+k,1)];
end
bottleconc{k}=bottleconcs;
end

Pcon=cell(3,1);%P concentration for interval, P* (P평균)
gr=cell(3,1);
for k=1:3
    nutmeanconc=[];
    grt=[];
for i =1:(numel(conc)+1)/4 -1 %acclimation period를 빼지 않고  시작부터 다 넣음// G.S, K.v의 경우 어짜피 플라팅해야하니 데이터가 비어 있어서 더 적은 수의 cellconc로 맞춤.
Nt1=bottlePnut{k}(i);
Nt2=bottlePnut{k}(i+1);
nutmeanconc= [nutmeanconc;(Nt2 - Nt1) / (log(Nt2 / Nt1))];%P*
%t2 - t1 == 1;
Ct1=bottleconc{k}(i);
Ct2=bottleconc{k}(i+1);
grt = [grt;(log(Ct2 / Ct1)) / (1)] ;
end
Pcon{k}= nutmeanconc;
gr{k}=grt;
end

gr=horzcat(gr{:});
Pcon=horzcat(Pcon{:});
bottleconc=horzcat(bottleconc{:});
bottlePnut=horzcat(bottlePnut{:});

%% only if there's missing data--P.t.
valid_rows = all(~isnan(Pcon), 2);
% NaN이 없는 행과 14번째 행을 합치기
Pcon= [Pcon(valid_rows, :)];%중간에 데이터 빠진 경우 실행
valid_rows = all(~isnan(gr), 2);
% NaN이 없는 행과 14번째 행을 합치기
gr= [gr(valid_rows, :)];
%%
figure; 
hold on;
plot(Pcon(startdate:end,:), gr(startdate:end,:), 'k.', 'MarkerSize', 18);

[p,S]= polyfit(Pcon(startdate:end,:), gr(startdate:end,:), 1); % 처음 두 날을 제외하고 회귀
x_val = linspace(min(min(Pcon(startdate:end,:))), max(max(Pcon(startdate:end,:))), 20); % x 값 범위 설정
f = polyval(p, x_val);
disp(['regression_equation :' sprintf('y = %.4fx + %.4f', p(1), p(2))])
[y_fit, delta] = polyval(p, Pcon(startdate:end,:), S);
SSresid = sum((gr(startdate:end,:) - y_fit).^2);
SStotal = (length(gr(startdate:end,:)) - 1) * var(gr(startdate:end,:));
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

% %위 식이 이상한 것 같다..앞에 영양염 농도 구하는 식으로 한 번 구해보자-오 같게 나옴
% p = polyfit(Pcon(startdate:end,:), gr(startdate:end,:), 1);
% f = polyval(p, Pcon(startdate:end,:));
% TSS = sum((gr(startdate:end,:) - mean(gr(startdate:end,:))).^2);
% RSS = sum((gr(startdate:end,:) - f).^2);
% R_squared = 1 - (RSS / TSS);

F_statistic = (rsq / (1 - rsq)) * ((length(gr(startdate:end,:)) - 2) / 1);
% p-value 계산 (fcdf 함수 사용)
p_value = 1 - fcdf(F_statistic, 1, length(gr(startdate:end,:)) - 2);
disp(['P-value: ', num2str(p_value)]);

plot(x_val, f, '-', 'color', 'k', 'LineWidth', 2);
yticks(-0.2:0.2:1.2); 
xticks(0:1:20);
if speciesname=="S. acuminata"
xlim([9 20])
ylim([-0.2 0.8])
elseif speciesname=="P. triestinum"
xlim([11 20])
ylim([-0.2 1.2])
end

ax=gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -0.2:0.1:1.2;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off;

xlabel('PO_{4} (\muM)');
ylabel('Growth rate (d^{-1})');
text(16, 0.1, sprintf('p <0.05'), 'fontsize', 8);
%% figure 2A; for NO3 and gr
bottleNnut=cell(3,1);
for k=1:3
    bottlenuts=[];
for i=1:(numel(N)+1)/4
  bottlenuts=[bottlenuts;N(4*(i-1)+k,1)];
end
  bottleNnut{k}=bottlenuts;
end

Ncon=cell(3,1);
gr=cell(3,1);
for k=1:3
    nutmeanconc=[];
    grt=[];
for i =1:(numel(conc)+1)/4 -1 %G.S, K.v의 경우 어짜피 플라팅해야하니 데이터가 비어 있어서 더 적은 수의 cellconc로 맞춤.
Nt1=bottleNnut{k}(i);
Nt2=bottleNnut{k}(i+1);
nutmeanconc= [nutmeanconc;(Nt2 - Nt1) / (log(Nt2 / Nt1))];
%t2 - t1 == 1;
Ct1=bottleconc(i,k);
Ct2=bottleconc(i+1,k);
grt = [grt;(log(Ct2 / Ct1)) / (1)] ;
end
Ncon{k}= nutmeanconc;
gr{k}=grt;
end

gr=horzcat(gr{:});
Ncon=horzcat(Ncon{:});%N mean conc
bottleNnut=horzcat(bottleNnut{:});
%% only if there's missing data--p.t.
idx = all(isnan(Ncon), 2);
valid_rows=find(~idx);
% NaN이 없는 행과 14번째 행을 합치기
Ncon= [Ncon(valid_rows, :)];%중간에 데이터 빠진 경우 실행
valid_rows = all(~isnan(gr), 2);
% NaN이 없는 행과 14번째 행을 합치기
gr= [gr(valid_rows, :)];
%%
Ncon_fixed = Ncon;
gr_fixed=gr;

% Ensure Ncon and gr are reshaped into vectors
NconVec = reshape(Ncon_fixed(startdate:end,:), [], 1);%acclimation period
grVec = reshape(gr_fixed(startdate:end,:), [], 1);
valid_rows = all(~isnan(NconVec), 2);
NconVec= [NconVec(valid_rows, :)];%get rid of NaN ones
grVec= [grVec(valid_rows, :)];

figure; 
hold on;

plot(NconVec, grVec, 'k.', 'MarkerSize', 15);

% Find b(1) as the maximum of all gr values
b(1) = max(grVec);

% Calculate differences from b(1)/2 for each gr value to find b(2)
differences = abs(grVec - b(1)/2);
[minValue, minIndex] = min(differences);
% Use the index to find corresponding Ncon value for b(2)
b(2) = NconVec(minIndex);
disp(['Ncon value closest to half of gr_max: ', num2str(b(2))]);
x = linspace(min(NconVec), max(NconVec), 1000);
% Define the model function
mmFun = @(b,x) b(1) .* (NconVec ./ (b(2) + NconVec));

% Provide an initial guess for the fitting parameters
initialGuess = [b(1), b(2)];

% Curve fitting using vectors
[beta,R,J,CovB,MSE] = nlinfit(NconVec, grVec, mmFun, initialGuess);
betaci = nlparci(beta,R,J);% 95% CI of coeffieicents

f= beta(1).*x./(beta(2)+x);
plot(x,f,'k-','LineWidth',2)

% Calculating R-squared
SSresid = sum(R.^2);
SStotal = (length(grVec)-1) * var(grVec);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

ssresnonlin=sum(R.^2); %둘 다 같음.r^2구하는 두 번째 방식.
sstotnonlin=sum((grVec-mean(grVec)).^2);
rsqrnonlin=1-(ssresnonlin/sstotnonlin);

% Print the equation of the fitted curve
disp(['Fitted curve equation: gr = ' num2str(beta(1)) ' * (Ncon / (' num2str(beta(2)) ' + Ncon))']);

% Additional plot formatting
yticks(-0.2:0.2:1.2); 
xticks(0:20:160);
xlim([0 110])
ylim([-0.2 1.2])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -0.2:0.1:0.8;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:5:160;
ax.Box = 'off';
xline(ax, ax.XLim(2))
yline(ax, ax.YLim(2))
set(ax.YAxis(1), 'TickDir', 'out');
set(ax.XAxis(1), 'TickDir', 'out');
set(gca, 'FontWeight', 'bold');

text(60, 0, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
hold off;

xlabel('NO_{3} (\muM)');
ylabel('Growth rate (d^{-1})');
%% figure 3A; N and uptake of N

upk=cell(3,1);%uptake rate, 참고로 두번째 바틀은 마지막날,그리고 전날 셀 카운트가 동일해지는 바람에 셀 증가량으로 나눠서 NaN이 들어가버림(나누기 0이 되어서 그런 거같음)
for k=1:3
 upkt=[];
for i =1:size(Ncon,1)-1
Nt1=Ncon(i,k);
Nt2=Ncon(i+1,k);
%t2 - t1 == 1;
Ct1=bottleconc(i,k);
Ct2=bottleconc(i+1,k);
upkt = [upkt;(-Nt2+Nt1)*10^3/((Ct2-Ct1)/log(Ct2/Ct1))] ;%since it's a reduction,-+. 10^3 곱하는 거 맞는지 확인
end
upk{k}=upkt;
end
upk=horzcat(upk{:});

upk_fixed=upk;
Ncon_fixed=Ncon;
%%
if speciesname == "S. acuminata"
figure; 
hold on;

% reshaped into vectors
upkVec = reshape(upk_fixed(startdate:end+eyy,:), [], 1);
NconVec = reshape(Ncon_fixed(startdate:end-1+eyy,:), [], 1);

valid_rows = all(~isnan(upkVec), 2);
upkVec= [upkVec(valid_rows, :)];
NconVec= [NconVec(valid_rows, :)];
%아웃라이어가 생깁니다..
% upkVec(22)=NaN;
% upkVec= [upkVec(all(~isnan(upkVec), 2), :)];
% NconVec=[NconVec(all(~isnan(upkVec), 2), :)];
%여기까지
plot(NconVec, upkVec, 'k.', 'MarkerSize', 18);

% Find b(1) as the maximum of all upk values
b(1) = max(upkVec);

differences = abs(upkVec - b(1)/2);
[minValue, minIndex] = min(differences);
% Use the index to find corresponding Ncon value for b(2)
b(2) = NconVec(minIndex);
disp(['Ncon value closest to half of upk_max: ', num2str(b(2))]);
x = linspace(min(NconVec), max(NconVec), 1000);
% Define the model function
mmFun = @(b,x) b(1) .* (NconVec ./ (b(2) + NconVec));

% Provide an initial guess for the fitting parameters
initialGuess = [b(1), b(2)];

% Curve fitting using vectors
[beta,R,J,CovB,MSE] = nlinfit(NconVec, upkVec, mmFun, initialGuess);
betaci = nlparci(beta,R,J);% 95% CI of coeffieicents

f=beta(1).*x./(beta(2)+x);
plot(x,f,'k-','LineWidth',2)

% Calculating R-squared
SSresid = sum(R.^2);
SStotal = (length(upkVec)-1) * var(upkVec);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

ssresnonlin=sum(R.^2); %둘 다 같음.r^2구하는 두 번째 방식.
sstotnonlin=sum((upkVec-mean(upkVec)).^2);
rsqrnonlin=1-(ssresnonlin/sstotnonlin);

% Print the equation of the fitted curve
disp(['Fitted curve equation: gr = ' num2str(beta(1)) ' * (Ncon / (' num2str(beta(2)) ' + Ncon))']);

% Additional plot formatting
yticks(-2:3:16); 
xticks(0:20:160);
xlim([0 100])
ylim([-2 13])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:1:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:5:160;
ax.Box = 'off';
xline(ax, ax.XLim(2))
yline(ax, ax.YLim(2))
set(ax.YAxis(1), 'TickDir', 'out');
set(ax.XAxis(1), 'TickDir', 'out');
set(gca, 'FontWeight', 'bold');

text(60, 0, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
hold off;
end

if speciesname == "P. triestinum"

figure; 
hold on;
%보통
% NaN 값을 포함하지 않는 데이터 선택
invalid_indices = [1 2;2 2;3 2;11 2];
% invalid_indices = [1 2;2 2;3 2];
Ncon_valid = Ncon;
upk_valid = upk;
for i=1:size(invalid_indices,1)
Ncon_valid(invalid_indices(i,1),invalid_indices(i,2))=NaN;
upk_valid(invalid_indices(i,1),invalid_indices(i,2))=NaN;
end

% 새로운 데이터로 선형 회귀 수행

upkVec = reshape(upk_valid(startdate:end+eyy,:), [], 1);
NconVec = reshape(Ncon_valid(startdate:end-1+eyy,:), [], 1);
valid_rows = all(~isnan(NconVec), 2);
NconVec= [NconVec(valid_rows, :)];%get rid of NaN ones
upkVec= [upkVec(valid_rows, :)];

%튀는 값 제외!
if speciesname=="P. triestinum"
NconVec=NconVec([1:18,20:29],1);
upkVec=upkVec([1:18,20:29],1);
else
NconVec=NconVec([1:15,17:24],1);
upkVec=upkVec([1:15,17:24],1);
end
plot(NconVec, upkVec, 'k.', 'MarkerSize', 18);%uptake의 경우 10까지만 가능
[p, S] = polyfit(NconVec, upkVec, 1);
x_val = linspace(min(NconVec), max(NconVec), 20); % x 값 범위 설정
f = polyval(p, x_val);
disp(['regression_equation :' sprintf('y = %.4fx + %.4f', p(1), p(2))])
[y_fit, delta] = polyval(p, NconVec, S);
SSresid = sum((upkVec - y_fit).^2);
SStotal = (length(upkVec) - 1) * var(upkVec);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

F_statistic = (rsq / (1 - rsq)) * ((length(upkVec) - 2) / 1);
% p-value 계산 (fcdf 함수 사용)
p_value = 1 - fcdf(F_statistic, 1, length(upkVec) - 2);
disp(['P-value: ', num2str(p_value)]);

plot(x_val, f, '-', 'color', 'k', 'LineWidth', 2);
text(70, -0.03, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
hold off;
xlabel('NO_{3} (\muM)');
ylabel('Uptake (pM cell^{-1} d^{-1})');

if speciesname=="P. triestinum"
yticks(-2:1:16); 
xticks(0:20:100);
xlim([0 100])
ylim([-1 7])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:0.5:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:5:160;
else
yticks(-2:3:16); 
xticks(0:20:100);
xlim([0 100])
ylim([-2 13])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:1:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:5:160;
end
ax.Box = 'off';
xline(ax, ax.XLim(2))
yline(ax, ax.YLim(2))
set(ax.YAxis(1), 'TickDir', 'out');
set(ax.XAxis(1), 'TickDir', 'out');
set(gca, 'FontWeight', 'bold');

end
%% P
upk=cell(3,1);%uptake rate, 참고로 두번째 바틀은 마지막날,그리고 전날 셀 카운트가 동일해지는 바람에 셀 증가량으로 나눠서 NaN이 들어가버림(나누기 0이 되어서 그런 거같음)
for k=1:3
 upkt=[];
for i =1:size(Pcon,1)-1 %N스타가 하나 적으니..13으로..
Nt1=Pcon(i,k);
Nt2=Pcon(i+1,k);
%t2 - t1 == 1;
Ct1=bottleconc(i,k);
Ct2=bottleconc(i+1,k);
upkt = [upkt;(-Nt2+Nt1)*10^3/((Ct2-Ct1)/log(Ct2/Ct1))] ;%since it's a reduction,-+. 10^3 곱하는 거 맞는지 확인
end
upk{k}=upkt;
end
upk=horzcat(upk{:});

%%
figure; 
hold on;
plot(Pcon(startdate:end-1+eyy,:), upk(startdate:end+eyy,:), 'k.', 'MarkerSize', 18);%uptake의 경우 10까지만 가능
%보통
[p,S]= polyfit(Pcon(startdate:end-1+eyy,:), upk(startdate:end+eyy,:), 1); % 처음 두 날을 제외하고 회귀
% % in the case of s.a. 14day bottle 2's uptake rate was NaN 그래서 polyfit시키려고 Pcon도 NaN함.
% indexes = true(14, 3); 
% indexes(14, 2) = false; 
% indexes(1:startdate-1, :) = false; 
% [p,S]= polyfit(Pcon(indexes), upk(indexes), 1);%여기까지가 s.a.의 경우

x_val = linspace(min(min(Pcon(startdate:end-1+eyy,:))), max(max(Pcon(startdate:end-1+eyy,:))), 20); % x 값 범위 설정
f = polyval(p, x_val);
disp(['regression_equation :' sprintf('y = %.4fx + %.4f', p(1), p(2))])
[y_fit, delta] = polyval(p, Pcon(startdate:end-1+eyy,:), S);
SSresid = sum((upk(startdate:end+eyy,:) - y_fit).^2);
SStotal = (length(upk(startdate:end+eyy,:)) - 1) * var(upk(startdate:end+eyy,:));
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

F_statistic = (rsq / (1 - rsq)) * ((length(upk(startdate:end+eyy,:)) - 2) / 1);
% p-value 계산 (fcdf 함수 사용)
p_value = 1 - fcdf(F_statistic, 1, length(upk(startdate:end+eyy,:)) - 2);
disp(['P-value: ', num2str(p_value)]);

plot(x_val, f, '-', 'color', 'k', 'LineWidth', 2);
hold off;

xlabel('PO_{4} (\muM)');
ylabel('Uptake (pM cell^{-1} d^{-1})');

if speciesname=="P. triestinum"
yticks(-0.1:0.2:0.7); 
xticks(0:1:25);
xlim([10 20])
ylim([-0.1 0.7])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:0.1:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:160;
text(17.3, 0.01, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
else
yticks(-0.1:0.1:0.7); 
xticks(0:1:25);
xlim([10 18])
ylim([0 0.6])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:0.05:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:160;
text(16, 0.05, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
end
ax.Box = 'off';
xline(ax, ax.XLim(2))
yline(ax, ax.YLim(2))
set(ax.YAxis(1), 'TickDir', 'out');
set(ax.XAxis(1), 'TickDir', 'out');
set(gca, 'FontWeight', 'bold');

%%
%혹시몰라서,,,그냥 N일 때 피겨 3
upk=cell(3,1);%uptake rate, 참고로 두번째 바틀은 마지막날,그리고 전날 셀 카운트가 동일해지는 바람에 셀 증가량으로 나눠서 NaN이 들어가버림(나누기 0이 되어서 그런 거같음)
for k=1:3
 upkt=[];
for i =1:size(bottleNnut,1)-1 %N스타가 하나 적으니..13으로..
Nt1=bottleNnut(i,k);
Nt2=bottleNnut(i+1,k);
%t2 - t1 == 1;
Ct1=bottleconc(i,k);
Ct2=bottleconc(i+1,k);
upkt = [upkt;(-Nt2+Nt1)*10^3/((Ct2-Ct1)/log(Ct2/Ct1))] ;%since it's a reduction,-+. 10^3 곱하는 거 맞는지 확인
end
upk{k}=upkt;
end
upk=horzcat(upk{:});

upk_fixed=upk;
Ncon_fixed=Ncon;
%%
figure; 
hold on;

% reshaped into vectors
upkVec = reshape(upk_fixed(startdate:end+eyy,:), [], 1);
NconVec = reshape(Ncon_fixed(startdate:end+eyy,:), [], 1);

valid_rows = all(~isnan(upkVec), 2);
upkVec= [upkVec(valid_rows, :)];
NconVec= [NconVec(valid_rows, :)];
%아웃라이어가 생깁니다..
% upkVec(22)=NaN;
% upkVec= [upkVec(all(~isnan(upkVec), 2), :)];
% NconVec=[NconVec(all(~isnan(upkVec), 2), :)];
%여기까지
plot(NconVec, upkVec, 'k.', 'MarkerSize', 18);

% Find b(1) as the maximum of all upk values
b(1) = max(upkVec);

differences = abs(upkVec - b(1)/2);
[minValue, minIndex] = min(differences);
% Use the index to find corresponding Ncon value for b(2)
b(2) = NconVec(minIndex);
disp(['Ncon value closest to half of upk_max: ', num2str(b(2))]);
x = linspace(min(NconVec), max(NconVec), 1000);
% Define the model function
mmFun = @(b,x) b(1) .* (NconVec ./ (b(2) + NconVec));

% Provide an initial guess for the fitting parameters
initialGuess = [b(1), b(2)];

% Curve fitting using vectors
[beta,R,J,CovB,MSE] = nlinfit(NconVec, upkVec, mmFun, initialGuess);
betaci = nlparci(beta,R,J);% 95% CI of coeffieicents

f=beta(1).*x./(beta(2)+x);
plot(x,f,'k-','LineJoin',2)

% Calculating R-squared
SSresid = sum(R.^2);
SStotal = (length(upkVec)-1) * var(upkVec);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

ssresnonlin=sum(R.^2); %둘 다 같음.r^2구하는 두 번째 방식.
sstotnonlin=sum((upkVec-mean(upkVec)).^2);
rsqrnonlin=1-(ssresnonlin/sstotnonlin);

% Print the equation of the fitted curve
disp(['Fitted curve equation: gr = ' num2str(beta(1)) ' * (Ncon / (' num2str(beta(2)) ' + Ncon))']);

% Additional plot formatting
yticks(-2:3:16); 
xticks(0:20:160);
xlim([0 160])
ylim([-2 13])
ax = gca;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = -2:1:16;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:5:160;
ax.Box = 'off';
xline(ax, ax.XLim(2))
yline(ax, ax.YLim(2))
set(ax.YAxis(1), 'TickDir', 'out');
set(ax.XAxis(1), 'TickDir', 'out');
set(gca, 'FontWeight', 'bold');

text(60, 0, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
hold off;

xlabel('NO_{3} (\muM)');
ylabel('Uptake (pM cell^{-1} d^{-1})');

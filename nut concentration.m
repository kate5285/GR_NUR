clc; clear all;
%%
% file_path = "D:\서울대\3-S internship\최종 결과값\Growth rate and nutrient uptake (Scrippsiella acuminata) 2022 0913 OJH.xlsx";
file_path ="D:\서울대\3-S internship\최종 결과값\Growth rate and nutrient uptake (Prorocentrum triestinum) 2022 0913 OJH.xlsx";
% file_path ="D:\서울대\3-S internship\최종 결과값\Growth rate and nutrient uptake (Gertia stigmatica) 2022 0913 OJH.xlsx";
% file_path ="D:\서울대\3-S internship\최종 결과값\Growth rate and nutrient uptake (Karlodinium veneficum) 2022 0913 OJH.xlsx";

% speciesname="S. acuminata";
speciesname="P. triestinum";
% speciesname="G. stigmatica";
% speciesname="K. veneficum";
GS_abundance= xlsread(file_path, 4, 'E2:E61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기-다 읽으려면 45 대신에 61
GS_abundance=rmmissing(GS_abundance);
fulldays=15;

if speciesname=="P. triestinum"
Ndays=14; % for case with missing data, in this case I put 14 since one day worth of data is missing
else
Ndays=fulldays; % for case when no data is missing
end
GS_abundance=reshape(GS_abundance, 3,Ndays);
GS_abundance=GS_abundance';

GS_N= xlsread(file_path, 4, 'K2:K61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기
GS_N=rmmissing(GS_N);
GS_N=reshape(GS_N, 3,Ndays);
GS_N=GS_N';

GS_PO= xlsread(file_path, 4, 'W2:W61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기
GS_PO=rmmissing(GS_PO);
GS_PO=reshape(GS_PO, 3,Ndays);
GS_PO=GS_PO';

GS_Si= xlsread(file_path, 4, 'AC2:AC61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기
GS_Si=rmmissing(GS_Si);
GS_Si=reshape(GS_Si, 3,Ndays);
GS_Si=GS_Si';

GS_NH4= xlsread(file_path, 4, 'Q2:Q61'); % 4번째 시트에서 W column의 2~45 row 데이터 읽기
GS_NH4=rmmissing(GS_NH4);
GS_NH4=reshape(GS_NH4, 3,Ndays);
GS_NH4=GS_NH4';

day = [0:fulldays-1];
dates = [0:fulldays-1,0:fulldays-1,0:fulldays-1];
dates = reshape(dates, fulldays, 3);
day = squeeze(day);
startdate=1;
enddate=6;
short_day=[startdate:enddate];
short_day=squeeze(short_day);
short_dates_index = [startdate:enddate,startdate:enddate,startdate:enddate];
short_dates_index = reshape(short_dates_index, enddate-startdate+1,3);

ln_gs=log(GS_abundance);
ln_gs_short_ver = ln_gs(startdate+1:enddate+1,:);
%% only do this for missing data--p.t.
variables = {GS_abundance, GS_N, GS_PO, GS_Si, GS_NH4};
missingdate=12; %빈 날짜
% 각 변수에 대해 재배열 작업 수행
for i = 1:numel(variables)
    % 변수의 원래 값을 저장
    variable_ori = variables{i};
    
    % NaN으로 채워진 1x3 크기의 배열 생성
    nan_row = NaN(1, 3);
    
    % 변수를 재배열하고 NaN 행을 삽입
    variables{i} = vertcat(variable_ori(1:missingdate, :), nan_row, variable_ori(missingdate+1:end, :));
end

% 결과를 다시 GS_N, GS_PO, GS_Si, GS_NH4 변수에 할당
[GS_abundance,GS_N, GS_PO, GS_Si, GS_NH4] = variables{:};
ln_gs=log(GS_abundance);
%% figures
% abundance
figure;
subplot(4,2,1); 
axis equal;
plot(day(1:15), GS_abundance, '.', 'MarkerSize', 15, 'color', 'k'); 
hold on;
ax = gca;
ylabel('Abundance (cells mL^{-1})','fontweight','bold');

xticks(0:2:16);
xlim([0 16])
if speciesname=="S. acuminata"
yticks(0:2000:65000); 
ylim([0 18000])
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = 0:1000:65000; 
text(8, 1000, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
elseif speciesname=="P. triestinum"
ylim([0 65000])
yticks(0:10000:65000); 
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = 0:5000:65000; 
text(9, 5000, sprintf('{\\it %s} LD', speciesname), 'fontsize', 8);
end

ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 

disp(max(GS_abundance(:)))

% log scale abundance all days
subplot(4,2,3);
axis equal;
plot(day(1:15), ln_gs, '.', 'MarkerSize', 15, 'color', 'k');
hold on;

valid_rows = all(~isnan(ln_gs), 2);
% NaN이 없는 행과 14번째 행을 합치기
ln_gs_fixed= [ln_gs(valid_rows, :)];%중간에 데이터 빠진 경우 실행
dates_fixed= [dates(valid_rows, :)];

ylabel('Ln (AB) (cells mL^{-1})','fontweight','bold');
[p,S] = polyfit(dates_fixed(:,:), ln_gs_fixed, 1);
f = polyval(p, dates_fixed(:,:)); 
plot(dates_fixed(:,1), f, '-', 'color', 'k', 'LineWidth', 1);

[y_fit, delta] = polyval(p,dates_fixed(:,:), S);
SSresid = sum((ln_gs_fixed - y_fit).^2);
SStotal = (length(ln_gs_fixed) - 1) * var(ln_gs_fixed);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

F_statistic = (rsq / (1 - rsq)) * ((length(ln_gs_fixed) - 2) / 1);
% p-value 계산 (fcdf 함수 사용)
p_value = 1 - fcdf(F_statistic, 1, length(ln_gs_fixed) - 2);
disp(['P-value: ', num2str(p_value)]);
disp(['regression_equation :' sprintf('y = %.4fx + %.4f', p(1), p(2))])

ax = gca;
yticks(4:1:12); 
xticks(0:2:16);
xlim([0 16])
if speciesname=="S. acuminata"
ylim([4 11])
elseif speciesname=="P. triestinum"
ylim([4 12])
end
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(ln_gs(:)))

% log scale short version days
subplot(4,2,5);
axis equal;
plot(short_day, ln_gs_short_ver, '.', 'MarkerSize', 15, 'color', 'k');
ax = gca;
hold on;
ylabel('Ln (AB) (cells mL^{-1})','fontweight','bold');
[p,S]= polyfit(short_dates_index, ln_gs_short_ver, 1);
f = polyval(p, short_dates_index); 
plot(short_day, f, '-', 'color', 'k', 'LineWidth', 1) ;

[y_fit, delta] = polyval(p,short_dates_index, S);
SSresid = sum((ln_gs_short_ver - y_fit).^2);
SStotal = (length(ln_gs_short_ver) - 1) * var(ln_gs_short_ver);
rsq = 1 - SSresid/SStotal;
disp(['R-squared: ', num2str(rsq)]);

F_statistic = (rsq / (1 - rsq)) * ((length(ln_gs_short_ver) - 2) / 1);
% p-value 계산 (fcdf 함수 사용)
p_value = 1 - fcdf(F_statistic, 1, length(ln_gs_short_ver) - 2);
disp(['P-value: ', num2str(p_value)]);
disp(['regression_equation :' sprintf('y = %.4fx + %.4f', p(1), p(2))])

yticks(4:1:12); 
xticks(0:2:16);
xlim([0 16])
ylim([4 12])
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(ln_gs_short_ver(:)))

% N concentration
subplot(4,2,2);
axis equal;

% 아예 빼버리기
day_modified = dates;
GS_N_modified = GS_N;
if speciesname=="S. acuminata"
    excluded_indices=([1 2; 3 2; 12 3]);
    % excluded_indices=([1 2; 3 2]);
elseif speciesname=="P. triestinum"
    excluded_indices=([3 2; 15 2]);
else
    disp("error")
end
for i=1:size(excluded_indices,1)
day_modified(excluded_indices(i,1),excluded_indices(i,2))=NaN;
GS_N_modified(excluded_indices(i,1),excluded_indices(i,2))=NaN;
end

plot(day_modified, GS_N_modified, '.', 'MarkerSize', 15, 'color', 'k');
 text(0.5, 20, 'NO_3', 'fontsize', 8);

yticks(0:20:220); % 빼버리는 경우이면 y축 범위를 축소시켜야 함.
xticks(0:2:16);
xlim([0 16])
if speciesname=="S. acuminata"
ylim([0 120])
elseif speciesname=="P. triestinum"
ylim([0 160])
end

ax=gca;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = 0:10:200;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(GS_N(:)))

% P
subplot(4,2,4);
axis equal;
plot(day, GS_PO, '.', 'MarkerSize', 15, 'color', 'k');
hold on;
ylabel('Concentration (\muM)','fontweight','bold');
text(0.5, 3, 'PO_4', 'fontsize', 8);
yticks(0:5:25); 
xticks(0:2:16);
xlim([0 16])
ylim([0 25])
ax=gca;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = 0:1:25;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(GS_PO(:)))

% ratio of N/P
ratio = GS_N ./ GS_PO;
subplot(4,2,6);
axis equal;
% 아예 빼버리기
ratio_modified = ratio;
for i=1:size(excluded_indices,1)
ratio_modified(excluded_indices(i,1),excluded_indices(i,2))=NaN;
end
plot(day_modified, ratio_modified, '.', 'MarkerSize', 15, 'color', 'k');
ylabel('Ratio NO_3/ PO_4','fontweight','bold');

yticks(0:1:12); %마찬가지로 뺴는 경우면 y축 범위를 조정해야함
xticks(0:2:16);
xlim([0 16])
ylim([0 8])
ax=gca;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(ratio(:)))

% Si
subplot(4,2,7);
axis equal;
plot(day, GS_Si, '.', 'MarkerSize', 15, 'color', 'k');
hold on;
xlabel('Elapsed time (d)','fontweight','bold');
ylabel('Concentration (\muM)','fontweight','bold');
yticks(0:5:40); 
xticks(0:2:16);
xlim([0 16.8])
if speciesname=="S. acuminata"
range=[15 25];
text_y=16;
elseif speciesname=="P. triestinum"
 text_y=26;
range=[15 40];
end
text(0.5, text_y, 'SiO_2', 'fontsize', 8);
ylim(range)
ax=gca;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:16;
ax.YAxis(1).MinorTick = 'on'; 
ax.YAxis(1).MinorTickValues = 0:1:40;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(GS_Si(:)))

% NH4
subplot(4,2,8);
axis equal;
plot(day, GS_NH4, '.', 'MarkerSize', 15, 'color', 'k');
hold on;
xlabel('Elapsed time (d)','fontweight','bold');
ylabel('Concentration (\muM)','fontweight','bold');
if speciesname=="S. acuminata"
ylim([0 12])
text(0.5, 1, 'NH_4', 'fontsize', 8);
elseif speciesname=="P. triestinum"
ylim([6 17])
text(0.5, 15, 'NH_4', 'fontsize', 8);
end
yticks(0:1:17); 
xticks(0:2:16);
xlim([0 16])
ax=gca;
ax.XAxis(1).MinorTick = 'on'; 
ax.XAxis(1).MinorTickValues = 0:1:17;
ax.Box = 'off';
xline(ax,ax.XLim(2))
yline(ax,ax.YLim(2))
set(ax.YAxis(1),'TickDir','out');
set(ax.XAxis(1),'TickDir','out');
set(gca,'FontWeight','bold');
hold off; 
disp(max(GS_NH4(:)))


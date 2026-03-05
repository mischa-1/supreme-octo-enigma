%% Load Data in

quantized_80 = readmatrix("quantized_80channels.csv");
quantized_3 = readmatrix("quantized_3channels.csv");

quantized_80 = quantized_80(2:end);
quantized_3  = quantized_3(2:end);

%% Get mean and standard deviation

mean_80 = mean(quantized_80, 1);
std_80 = std(quantized_80, 0, 1);
mean_3 = mean(quantized_3, 1);
std_3 = std(quantized_3, 0, 1);

ymax = prctile(all_data,95);  % ignore extreme 5%

%% Organize & Label

means = [mean_80 mean_3];
stds = [std_80 std_3];
labels = ["80 COCO Channels", "3 Self-Trained Channels"];

%% Graph

figure
b = bar(means, 'FaceColor','flat');
hold on

% nicer colors
b.CData(1,:) = [0.2 0.6 0.8];
b.CData(2,:) = [0.9 0.4 0.4];

% error bars
er = errorbar(1:2, means, stds, ...
    'k', 'LineStyle','none', 'LineWidth',1.5);

% axis formatting
set(gca,'XTickLabel',labels)
ylabel("Time per Frame (s)")
title("YOLO Inference Time Comparison")

grid on
box off

% improve readability
set(gca,'FontSize',12)
ylim([0 max(means+stds)*1.2])

hold off

%% Scatter

figure
hold on

% scatter points with jitter
scatter(ones(size(quantized_80))*1 + randn(size(quantized_80))*0.02,...
        quantized_80,40,'filled')

scatter(ones(size(quantized_3))*2 + randn(size(quantized_3))*0.02,...
        quantized_3,40,'filled')

% mean lines
plot([0.8 1.2],[mean(quantized_80) mean(quantized_80)],'k','LineWidth',3)
plot([1.8 2.2],[mean(quantized_3) mean(quantized_3)],'k','LineWidth',3)

set(gca,'XTick',[1 2])
set(gca,'XTickLabel',["80 COCO Channels","3 Self-Trained Channels"])

ylabel("Time per Frame (s)")

title("YOLO Inference Time Distribution")

grid on
box off
set(gca,'FontSize',12)

%% Box Plot

data = [quantized_80 quantized_3];
groups = [ones(length(quantized_80),1); 2*ones(length(quantized_3),1)];

values = [quantized_80; quantized_3];

figure

boxchart(groups,values,'BoxWidth',0.5)

set(gca,'XTick',[1 2])
set(gca,'XTickLabel',["80 COCO Channels","3 Self-Trained Channels"])

ylabel("Time per Frame (s)")
title("YOLO Inference Time Distribution")

grid on
box off
set(gca,'FontSize',12)

%% Publication Level


figure
hold on

% jitter scatter
scatter(1 + randn(size(quantized_80))*0.02,quantized_80,40,...
       [0.2 0.6 0.8],'filled','MarkerFaceAlpha',0.6)

scatter(2 + randn(size(quantized_3))*0.02,quantized_3,40,...
       [0.9 0.4 0.4],'filled','MarkerFaceAlpha',0.6)

% mean and std
errorbar(1,mean(quantized_80),std(quantized_80),'k','LineWidth',2)
errorbar(2,mean(quantized_3),std(quantized_3),'k','LineWidth',2)

scatter(1,mean(quantized_80),120,'k','filled')
scatter(2,mean(quantized_3),120,'k','filled')

set(gca,'XTick',[1 2])
set(gca,'XTickLabel',["80 COCO Channels","3 Self-Trained Channels"])

ylabel("Time per Frame (s)")
title("YOLO Model Inference Time Comparison")

grid on
box off
set(gca,'FontSize',12)
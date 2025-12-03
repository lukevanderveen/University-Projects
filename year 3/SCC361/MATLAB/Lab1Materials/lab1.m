tab = readtable('coronavirus-cases.csv');
x = tab(1:10, :);
%x

rowsID = strcmp(tab.AreaName, 'England');
y = tab(rowsID, :);
%y

sorted = sortrows(tab, "SpecimenDate");
z = sorted(1:10, :);
%z

tab.SpecimenDate = datetime(tab.SpecimenDate);
tab = unique(tab); %only use unique entries
tab = rmmissing(tab); %remove missing date or cases entries
figure(1);
plot(tab.SpecimenDate, tab.DailyCases, 'b');
title('Daily cases against Specimen Date');
xlabel('Date');
ylabel('Number of Cases');
grid("on");

tab.SevenDayAverage = zeros(height(tab), 1);
for i = 7:height(tab)
       tab.SevenDayAverage(i) = mean(tab.DailyCases(i-6:i));
end

figure(2);
plot(tab.SpecimenDate, tab.DailyCases, 'b');
hold on;
plot(tab.SpecimenDate, tab.SevenDayAverage, 'r');
legend('daily cases', '7-day average');
title('Daily cases against Specimen Date with 7 Day Average');
xlabel('Date');
ylabel('Number of Cases');
grid("on");
hold off;
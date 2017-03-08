demand=textread('hw2_max_demand.txt');
temp=textread('hw2_max_temp.txt');
y=demand;x=temp;
%normalize-data range 0~1
demand = (demand - mean(demand))/range(demand);
temp= (temp-mean(temp))/range(temp);
a=length(demand);
mu = 0.1;
lambda =0;
index=round(0.7*a);
Phi = [ones(index,1),temp(1:index,1)];
% linear regression
theta = randn(2,1); % initial-theta
gtheta = 1;
Phi'*(Phi*theta-demand(1:index,1))
while mean(abs(gtheta)) > 1e-5 % iteration stop constraint
 gtheta = Phi'*(Phi*theta-demand(1:index,1))/index+lambda*[0;theta(2:end)];
 theta = theta-mu*gtheta;
 cost_function = (Phi*theta-demand(1:index,1))'*(Phi*theta-
demand(1:index,1))/(2*index) + lambda/2*theta(2:end).'*theta(2:end);
end


figure(2)
plot(x,y,'*');
hold on
temp_in =[0:0.1:2]';
demand_est = [ones(length(temp_in),1),temp_in]*theta;
plot(temp_in*range(x),demand_est*normalize_demand ,'r-');
axis([0, 100, 1.2, 3.2])

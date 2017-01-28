function [] = main()

load('catdat1.mat');

% initialize values

m = 100; % batch size
alpha = 3.0; % learning rate

w{2} = (rand(30,784)-0.5); % weight matrix in second layer
w{3} = (rand(13,30)-0.5); % weight matrix in third layer

b{2} = -(rand(30,1)-0.5); % bias matrix in second layer
b{3} = -(rand(13,1)-0.5); % bias matrix in third layer

counter = 1;
cost{1}(1) = 100; % initialize cost to 100 just to get into while loop
while sum(cost{counter}) > 3
	counter = counter + 1;
	% get mini batch
	[batch,batchnum] = getbatch(m,data,num);
	assignin('base','batch',batch);
	assignin('base','batchnum',batchnum);
	% for each training pair in batch
	for i = 1:m
		
		% set input activation
		for j = 1:784
			a{i}{1} = batch{i};
		end

		% feedforward
		for j = 2:3
			z{i}{j} = w{j}*a{i}{j-1}+b{j};
			a{i}{j} = sigmoid(z{i}{j});
		end

		% output error
		cost{counter}(i) = sum(calccost(a{i}{3},batchnum{i}));
		d{i}{3} = calcdcost(a{i}{j},batchnum{i}) .* sigmoid_prime(z{i}{3});
		
		% backpropagate the error
		for j = 2:2
			d{i}{j} = w{j+1}'*d{i}{j+1} .* sigmoid_prime(z{i}{j});
		end
	end

	assignin('base','d',d);
	assignin('base','a',a);
	for j = 3:-1:2

		wup{j} = 0;
		bup{j} = 0;

		for i = 1:m 
			wup{j} = wup{j} + d{i}{j} * a{i}{j-1}';
			bup{j} = bup{j} + d{i}{j};
		end

		w{j} = w{j} + (alpha/m) * wup{j};
		b{j} = b{j} + (alpha/m) * bup{j};

	end

	% check accuracy
	correct = 0;
	for i = 1:m
		guessnum = find(a{i}{3}==max(a{i}{3}));
		correctnum = find(batchnum{i}==max(batchnum{i}));
		if guessnum==correctnum
			correct = correct + 1;
		end
	end

	disp(sum(cost{counter}))
	disp(correct)
	assignin('base','out',a{1}{3})
	assignin('base','batchnum',batchnum{1})

	

end

A = who;
for i = 1:length(A)
	assignin('base', A{i}, eval(A{i}));
end

save weightdat.mat  w  b

end

%--------------------------------------------------------------------------------------------%

function [batch,batchnum] = getbatch(m,data,num)
	ind = [ceil(51500*rand(1,m))];
	batch = data(ind);
	batchnum = num(ind);
end

function [out1] = sigmoid(x)
	out1 = 1./(1+exp(-x));
end

function [out1] = sigmoid_prime(x)
	out1 = (1-1./(1+exp(-x))) .* (1./(1+exp(-x)));
end

function [out1] = calccost(x1,x2)
	% x1 actual, x2 desired
	out1 = 0.5.*(x1-x2).^2;
end

function [out1] = calcdcost(x1,x2)
	out1 = x2-x1;
end
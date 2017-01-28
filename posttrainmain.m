%%posttrainmain

load('weightdat.mat')
load('catdat1.mat')
A=load('catdat1.mat')
% so now i need to propagate an image so okay
firsttest=A.data{50001,1}%ifromdraw{1}%images{51400}

a{1}{1}=firsttest

number=[0 1 2 3 4 5 6 7 8 9  11 12 13]


for j = 2:3
			z{1}{j} = w{j}*a{1}{j-1}+b{j};
			a{1}{j} = sigmoid(z{1}{j});
end

number(find( (a{1}{3})==max(a{1}{3})))


function [out1] = sigmoid(x)
	out1 = 1./(1+exp(-x));
end



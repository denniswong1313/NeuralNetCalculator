A=load('data1.mat')
B=[load('trainzero1.mat') load('traintwo2.mat') load('trainthree3.mat') load('trainplus3')  load('trainminus5.mat') load('trainmult4.mat') ]

%selen=length(A)
%imshow(1-(max(ifromdraw{1})-ifromdraw{1}))
firlen=length(A.num)

for t=1:length(A.images)
A.data{t}=round(1-A.data{t});%(max(A.images{t})-A.images{t});
end

for t=1:length(A.images)
A.images{t}=reshape(round(1-A.images{t}),[784 1]);%(max(A.images{t})-A.images{t});
end


for j=1:length(B)
    
for i=1:length(B(j).num)
   A.num{firlen+i    }=single(B(j).num{i}(:))
   A.images{firlen+i  }=single(reshape(B(j).images{i},[784 1]))   
   A.data{firlen+i,2}=single(B(j).num{i}(:))
   A.data{firlen+i,1}=single(reshape(B(j).images{i},[784 1]))
end
firlen=firlen+length(B(j).num)
end

data=A.data(50001:end);
images=A.images(50001:end);
num=A.num(50001:end);
seclen=length(A)
save catdat1.mat data images num



save catdat1.mat data images num

size(num)
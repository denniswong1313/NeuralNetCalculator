%function [] = realmain()

load('weightdat.mat')
load('catdat1.mat')
%dat1=re
%posttrainmain2(imageex,w)
ifromdraw=drawCharacter()
wg=load('weightdat.mat')
num1=posttrainmain2(ifromdraw{1},wg)
num2=posttrainmain2(ifromdraw{2},wg)
num3=posttrainmain2(ifromdraw{3},wg)





%end

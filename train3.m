function picdata=train(trainnumber,character)



%imagedata{1,2}{3}=[0 0 ]
im_size = 28; %size of the final image
input_num = trainnumber; %number of inputs  this is how many pictures you want to make you can only do one type, +,-,= at a time
scale_factor = 100;
imagedata={{1},{1}}


placevector=[0;0;0;0;0;0;0;0;0;0;0;0;0]
switch character
    
     case 0
        placevector(1)=1
    
     case 1
        placevector(2)=1
    case 2
        placevector(3)=1
    case 3
        placevector(4)=1
    
    case 4
        placevector(5)=1
    case 5
        placevector(6)=1
    case 6
        placevector(7)=1
    
    case 7
        placevector(8)=1
    case 8
        placevector(9)=1
    case 9
        placevector(10)=1
    
    case '+'
        placevector(11)=1
    case '-'
        placevector(12)=1
    case '*'
        placevector(13)=1
    %case '='
     %   placevector(14)=1
       
end
        
    

for a = 1:input_num
    image{a} = ones(im_size);
    
    % Opens a blank image and allows user to draw. Press 'Escape' when
    % finished.
    figure
    imshow(ones(im_size*scale_factor));
    h = imfreehand('Closed', false); %open drawing canvas
    lastim = h;
    while ~isempty(lastim)
        lastim = imfreehand('Closed', false);
        h = cat(1,h,lastim);
    end
    h = h(isvalid(h));
    
    for i = 1:size(h,1)
        data = get(h(i));
        xydata = get(data.Children(4));
        x = xydata.XData;
        y = xydata.YData;
        t = 1:length(x);
        t_new = 1:(1/scale_factor):length(t);
        temp_x = interp1(t,x,t_new,'spline');  %interpolate a point in between for smoother lines
        temp_y = interp1(t,y,t_new,'spline');
        x_new{i} = temp_x;
        y_new{i} = temp_y;
    end
    
    close
    
    %replace white pixels with black ones based on the shape drawn
    for k = 1:size(x_new,2)
        temp_x = x_new{k};
        temp_y = y_new{k};
        [coor_imagex,coor_imagey] = meshgrid(temp_x, temp_y);
        for b = 1:length(temp_x)
            for c = 1:length(temp_y)
                if (coor_imagex(b,c) == temp_x(b) && coor_imagey(c,b) == temp_y(c))
                    image{a}(round(temp_y(c)/scale_factor),round(temp_x(b)/scale_factor)) = 0;
                    if round(temp_y(c)/scale_factor) < size(image{a},1)
                        image{a}((round(temp_y(c)/scale_factor)+1),round(temp_x(b)/scale_factor)) = 0;
                    end
                    if round(temp_x(c)/scale_factor) < size(image{a},1)
                        image{a}(round(temp_y(c)/scale_factor),(round(temp_x(b)/scale_factor)+1)) = 0;
                    end
                end
            end
        end
    end
    images2{a}=image{a}
    num2{a}=placevector
end

figure;
subplot(2,2,1)
imshow(image{1}(:));
%newfile=[filename, '.mat']
data=[images2;num2]'
images=images2
num=num2
save trainmult-D.mat data  images  num
end
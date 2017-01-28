function image = drawCharacter()

%% Instructions
% Draw the character you want. Press 'Escape' when finished.
%
% Returns a 1x3 cell of the input characters that the user has drawn.
% Also begins plotting the equation that is to be evaluated using a 1x5
% subplot. In order to continue editing this plot, use figure(2).


im_size = 28; %size of the final image
input_num = 1; %number of inputs (including operator)
scale_factor = 10;

for a = 1:input_num
    image{a} = ones(im_size);
end

%make the equal sign image
eq_sign = ones(im_size);
eq_sign(9,4:24) = 0;
eq_sign(19,4:24) = 0;

for a = 1:input_num
    clearvars -except image im_size scale_factor input_num a eq_sign
    
    % Opens a blank image and allows user to draw.
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
                    %The following makes the lines thicker. Comment out for
                    %first version.
                    if round(temp_y(c)/scale_factor) < size(image{a},1)
                        image{a}((round(temp_y(c)/scale_factor)+1),round(temp_x(b)/scale_factor)) = 0;
                    end
                    if round(temp_x(b)/scale_factor) < size(image{a},1)
                        image{a}(round(temp_y(c)/scale_factor),(round(temp_x(b)/scale_factor)+1)) = 0;
                    end
                end
            end
        end
    end
%     figure(2)
%     subplot(1,(input_num+2),a);
%     imshow(image{a});
%     if a == 3
%         subplot(1,(input_num+2),(a+1))
%         imshow(eq_sign)
%     end
end
end
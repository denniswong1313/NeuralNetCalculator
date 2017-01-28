function output_dec = NNmath(in1,op,in2)
% Training Neural Network
% Read in summands here
summand_1 = in1;
summand_2 = in2;

if(strcmp(op,'plus') || strcmp(op,'+'))
    op = 'add';
elseif(strcmp(op,'minus') || strcmp(op,'-'))
    op = 'sub';
elseif(strcmp(op,'times') || strcmp(op,'*'))
    op = 'mult';
end

alpha = 0.5;
epoch = 0;
neg = 0;

if (strcmp(op,'add') == 1)
    atraining_data{1,1} = [0 0 0 -1];
    atraining_data{2,1} = [1 0 0 -1];
    atraining_data{3,1} = [0 1 0 -1];
    atraining_data{4,1} = [1 1 0 -1];
    atraining_data{5,1} = [0 0 1 -1];
    atraining_data{6,1} = [1 0 1 -1];
    atraining_data{7,1} = [0 1 1 -1];
    atraining_data{8,1} = [1 1 1 -1];

    atraining_data{1,2} = [0 0];
    atraining_data{2,2} = [1 0];
    atraining_data{3,2} = [1 0];
    atraining_data{4,2} = [0 1];
    atraining_data{5,2} = [1 0];
    atraining_data{6,2} = [0 1];
    atraining_data{7,2} = [0 1];
    atraining_data{8,2} = [1 1];

elseif (strcmp(op,'sub') == 1)
    atraining_data{1,1} = [0 0 0 -1];
    atraining_data{2,1} = [1 0 0 -1];
    atraining_data{3,1} = [0 1 0 -1];
    atraining_data{4,1} = [1 1 0 -1];
    atraining_data{5,1} = [0 0 1 -1];
    atraining_data{6,1} = [1 0 1 -1];
    atraining_data{7,1} = [0 1 1 -1];
    atraining_data{8,1} = [1 1 1 -1];

    atraining_data{1,2} = [0 0];
    atraining_data{2,2} = [1 0];
    atraining_data{3,2} = [1 1];
    atraining_data{4,2} = [0 0];
    atraining_data{5,2} = [1 1];
    atraining_data{6,2} = [0 0];
    atraining_data{7,2} = [0 1];
    atraining_data{8,2} = [1 1];
elseif (strcmp(op,'mult') == 1)
else
    disp('ERROR - Operation not defined');
    return;
end

if (strcmp(op,'sub') == 1 || strcmp(op,'add') == 1)
    for i = 1:8
        atraining_data{i,3} = [0 0];
    end

    e_record = ones(8,2);
    e_squared = (e_record).^2;
    sum_squared = sum(e_squared(:)); 

        rand_min = -0.4;
        rand_max = 0.8;
        w14 = rand_min + rand*(rand_max-rand_min);
        w24 = rand_min + rand*(rand_max-rand_min);
        w34 = rand_min + rand*(rand_max-rand_min);
        w15 = rand_min + rand*(rand_max-rand_min);
        w25 = rand_min + rand*(rand_max-rand_min);
        w35 = rand_min + rand*(rand_max-rand_min);
        w16 = rand_min + rand*(rand_max-rand_min);
        w26 = rand_min + rand*(rand_max-rand_min);
        w36 = rand_min + rand*(rand_max-rand_min);
        th4 = rand_min + rand*(rand_max-rand_min);
        th5 = rand_min + rand*(rand_max-rand_min);
        th6 = rand_min + rand*(rand_max-rand_min);

        layer1 = [w14 w15 w16; w24 w25 w26; w34 w35 w36; th4 th5 th6];

        w47 = rand_min + rand*(rand_max-rand_min);
        w57 = rand_min + rand*(rand_max-rand_min);
        w67 = rand_min + rand*(rand_max-rand_min);
        w48 = rand_min + rand*(rand_max-rand_min);
        w58 = rand_min + rand*(rand_max-rand_min);
        w68 = rand_min + rand*(rand_max-rand_min);
        th7 = rand_min + rand*(rand_max-rand_min);
        th8 = rand_min + rand*(rand_max-rand_min);

        layer2 = [w47 w48; w57 w58; w67 w68; th7 th8];
    

    training_cy = 1;
    training_order = randperm(8);
    train_id = training_order(training_cy);

    while(sum_squared > .001)
        training_input = atraining_data{train_id,1};
        y_desired = atraining_data{train_id,2};
        y_L1_actual = sigmf(training_input*layer1,[1 0]);
        y_L1_actual = [y_L1_actual -1];
        y_actual = sigmf(y_L1_actual * layer2,[1 0]);

        e = y_desired - y_actual;
        e_record(train_id,:) = e;
        egrad7 = y_actual(1)*(1-y_actual(1))*e(1);
        egrad8 = y_actual(2)*(1-y_actual(2))*e(2);
        dw47 = alpha * egrad7 * y_L1_actual(1);
        dw57 = alpha * egrad7 * y_L1_actual(2);
        dw67 = alpha * egrad7 * y_L1_actual(3);
        dth7 = alpha * -1 * egrad7;
        dw48 = alpha * egrad8 * y_L1_actual(1);
        dw58 = alpha * egrad8 * y_L1_actual(2);
        dw68 = alpha * egrad8 * y_L1_actual(3);
        dth8 = alpha * -1 * egrad8;

        egrad4 = y_L1_actual(1) * (1-y_L1_actual(1)) * (egrad7 * w47 + egrad8 * w48);
        egrad5 = y_L1_actual(2) * (1-y_L1_actual(2)) * (egrad7 * w57 + egrad8 * w58);
        egrad6 = y_L1_actual(3) * (1-y_L1_actual(3)) * (egrad7 * w67 + egrad8 * w68);

        dw14 = alpha * egrad4 * training_input(1);
        dw24 = alpha * egrad4 * training_input(2);
        dw34 = alpha * egrad4 * training_input(3);
        dth4 = alpha * -1 * egrad4;
        dw15 = alpha * egrad5 * training_input(1);
        dw25 = alpha * egrad5 * training_input(2);
        dw35 = alpha * egrad5 * training_input(3);
        dth5 = alpha * -1 * egrad5;
        dw16 = alpha * egrad6 * training_input(1);
        dw26 = alpha * egrad6 * training_input(2);
        dw36 = alpha * egrad6 * training_input(3);
        dth6 = alpha * -1 * egrad6;

        w14 = w14 + dw14;
        w24 = w24 + dw24;
        w34 = w34 + dw34;
        w15 = w15 + dw15;
        w25 = w25 + dw25;
        w35 = w35 + dw35;
        w16 = w16 + dw16;
        w26 = w26 + dw26;
        w36 = w36 + dw36;
        w47 = w47 + dw47;
        w48 = w48 + dw48;
        w57 = w57 + dw57;
        w58 = w58 + dw58;
        w67 = w67 + dw67;
        w68 = w68 + dw68;
        th4 = th4 + dth4;
        th5 = th5 + dth5;
        th6 = th6 + dth6;
        th7 = th7 + dth7;
        th8 = th8 + dth8;

        layer1 = [w14 w15 w16; w24 w25 w26; w34 w35 w36; th4 th5 th6];
        layer2 = [w47 w48; w57 w58; w67 w68; th7 th8];
        e_squared = (e_record).^2;
        sum_squared = sum(e_squared(:));
        atraining_data{train_id,3}= y_actual;

        epoch = epoch + 1;
        training_cy = training_cy + 1;


        if (training_cy > 8)
            training_cy = 1;
            training_order = randperm(8);
        end

       train_id = training_order(training_cy);

        if (mod(epoch,30000) == 0)
            epoch
            sum_squared
        end
    end
end
    
    % Adding / Subtracting numbers
if (strcmp(op,'sub') == 1 || strcmp(op,'add') == 1)
    if((strcmp(op,'sub')==1) && (summand_1 < summand_2))
        tmp = summand_1;
        summand_1 = summand_2;
        summand_2 = tmp;
        neg = 1;
    end

    binary_in1 = (de2bi(summand_1,5,'left-msb'));
    binary_in2 = (de2bi(summand_2,5,'left-msb'));
    carry = 0;
    output = [0 0 0 0 0];

    for i = 1:5
        input_data = [binary_in1(6-i) binary_in2(6-i) carry -1];
        output_bit = sigmf(input_data*layer1,[1 0]);
        output_bit = [output_bit -1];
        output_bit = sigmf(output_bit*layer2,[1 0]);
        output(6-i) = output_bit(1);
        carry = output_bit(2);
    end

    for j = 1:5
        if output(j) > .95
            output(j) = 1;
        end
        if output(j) < .1
            output(j) = 0;
        end
    end

    output
    output_dec = bi2de(output,'left-msb');
    if neg == 1
        output_dec = -output_dec
        output = -output;
    else
        output_dec
    end
end

if (strcmp(op,'mult')==1)
    alpha = 0.9;
    epoch  = 0;
    for i = 1:10
        for j = 1:10
            mtraining_data{j+(i-1)*10,1} = [de2bi((i-1),4,'left-msb') de2bi((j-1),4,'left-msb') -1];
            mtraining_data{j+(i-1)*10,2} = [de2bi((i-1)*(j-1),8,'left-msb')];
        end
    end

    e_record = ones(100,7);
    e_squared = (e_record).^2;
    sum_squared = sum(e_squared(:));

    done = 0;
            
    slp0(1,1) = -0.4 + rand*(-0.2-(-0.4));
    slp0(2,1) = -0.3 + rand*(-0.1-(-0.3));
    slp0(3,1) = -0.3 + rand*(-0.1-(-0.3));
    slp0(4,1) = 11 + rand*(12-11);
    slp0(5,1) = -0.4 + rand*(-0.1-(-0.4));
    slp0(6,1) = -0.3 + rand*(-0.1-(-0.3));
    slp0(7,1) = -0.3 + rand*(-0.1-(-0.3));
    slp0(8,1) = 11 + rand*(12-11);
    slp0(9,1) = 16.7 + rand*(17.2-16.7);
    
    slp6(1,1) = 9 + rand*(10-9);
    slp6(2,1) = -2.5 + rand*(-1.5-(-2.5));
    slp6(3,1) = -2.5 + rand*(-1.5-(-2.5));
    slp6(4,1) = -.23 + rand*(-.15-(-.23));
    slp6(5,1) = 9 + rand*(10-9);
    slp6(6,1) = -2.3 + rand*(-1.7-(-2.3));
    slp6(7,1) = -2.3 + rand*(-1.7-(-2.3));
    slp6(8,1) = -.2 + rand*(-.15-(-.2));
    slp6(9,1) = 14.2 + rand*(14.7-14.2);
    
    hla = [0.6,-.3,.1,.1,.1,.1,1,-.1; ...
           0.1,-.5,.1,.1,-.1,-.1,.1,.1;...
           -10,-1,3,-4,-2,-2,-4,-1;...
           -6,-2,-4,1,-2,-3,-4,-4;...
           .1,-1,.2,.4,-.2,-.3,1.3,-.4;...
           .1,-.3,.1,.01,-.1,.1,.1,.2;...
           -4,.1,-5,-5,-1,-1,-4,-1;...
           2,-2,-10,-1,-2,-2,-4,-3;...
           -5,1,-4,-1,-1,-1,-14,-1];
       
    hlb = [0.1,0.9,0.2,0.5,-0.1,-1,0.5,1; ...
           1,-8,0.2,-10,-2,1,-7,-1; ...
           -6,-4,-5,-2,3,1,3,-5; ...
           -1,-10,-6,3,-9,-4,6,1;...
           1,0.1,-0.2,1,1,1,-2,-0.5;...
           1,2,1,-11,-9,-2,-0.5,-10;...
           -6,-4,0.5,-3,3,-6,3,-4;...
           -0.5,-0.5,-5,4,7,-8,-7,-8;...
           -6,-4,-0.5,1,-7,-2,-3,-6];
       
    hlc = [-7,3,-7,-8,3,-8,-5,0.2;...
           -0.5,-3,-3,-15,-1,0.1,0.5,2;...
           -5,-3,-3,-10,-5,4,-9,1;...
           -11,-7,3,3,-4,-4,-3,-4;...
           5,3,-9,-13,-4,-9,-6,4;...
           -0.2,-3,-1,-3,1,-13,-0.3,-4;...
           -7,-3,-0.5,3,-4,-9,-9,-4;...
           -4,-7,4,-7,-7,4,-4,-4;...
           -2,-22,-5,-7,1,-1,-4,-1];
       
    hld = [-2,-4,1,-3,-3,2,-9,-0.1;...
           5,-2,-3,0.1,-9,-2,-3,1;...
           -5,-2,-2,3,-2,-5,6,-4;...
           -3,2,1,-4,-3,2,4,-2;...
           -2,-4,-1,-3,-4,2,-9,1;...
           4,-8,-0.2,1,-1,-1,-4,-6;...
           -5,-2,-3,3,-3,-5,6,-3;...
           -3,-3,-2,-3,2,1,4,0.1;...
           -6,-6,-0.2,4,-5,-0.3,5,-1];
       
    hle = [-3,-5,-1,-1,-2,-1,-8,-2;...
           -0.1,0.2,-1,-1,-6,-2,3,-5;...
           -1,-2,-0.5,-0.5,1,-0.5,-5,1;...
           -0.5,-1,-0.5,-0.1,0.5,0.5,-3,0.5;...
           -3,-5,-1,-1,-2,-0.5,-8,-2;...
           -0.3,0.3,-0.5,-1,-6,-2,3,-5;...
           -1,-2,-0.5,-0.5,1.5,0.1,-5,1;...
           -1,-1,-0.3,-0.3,0.3,-0.2,-3,1;...
           -1,-3,0.3,-0.2,-4,-1,-3,-5];
    
    mlp1(1,1) = -13.2 + rand*(-12.9-(-13.2));
    mlp1(2,1) = -.8 + rand*(-.6-(-.8));
    mlp1(3,1) = -13.5 + rand*(-13.2-(-13.5));
    mlp1(4,1) = -4.6 + rand*(-4.2-(-4.6));
    mlp1(5,1) = -2.4 + rand*(-2.2-(-2.4));
    mlp1(6,1) = -2.6 + rand*(-2.4-(-2.6));
    mlp1(7,1) = 14.5 + rand*(14.8-14.5);
    mlp1(8,1) = -3.7 + rand*(-3.4-(-3.7));
    mlp1(9,1) = 6.5 + rand*(6.9-6.5);
    
    mlp2(1,1) = 12.3 + rand*(12.5-12.3);
    mlp2(2,1) = -14.8 + rand*(-14.6-(-14.8));
    mlp2(3,1) = -6.4 + rand*(-6.2-(-6.4));
    mlp2(4,1) = -20.5 + rand*(-20.1-(-20.5));
    mlp2(5,1) = 12.2 + rand*(12.4-12.2);
    mlp2(6,1) = -7.6 + rand*(-7.2-(-7.6));
    mlp2(7,1) = 12 + rand*(12.5-12);
    mlp2(8,1) = -13.4 + rand*(-13-(-13.4));
    mlp2(9,1) = 16.7 + rand*(17.2-16.7);
    
    mlp3(1,1) = -14.1 + rand*(-13.7-(-14.1));
    mlp3(2,1) = 11.3 + rand*(11.6-11.3);
    mlp3(3,1) = 9.5 + rand*(9.9-9.5);
    mlp3(4,1) = -18.7 + rand*(-18.2-(-18.7));
    mlp3(5,1) = -6.9 + rand*(-6.5-(-6.9));
    mlp3(6,1) = -15.9 + rand*(-15.5-(-15.9));
    mlp3(7,1) = -12.5 + rand*(-12-(-12.5));
    mlp3(8,1) = 8.3 + rand*(8.7-8.3);
    mlp3(9,1) = 15.3 + rand*(15.7-15.3);
    
    mlp4(1,1) = -12.9 + rand*(-12.5-(-12.9));
    mlp4(2,1) = -11.9 + rand*(-11.5-(-11.9));
    mlp4(3,1) = -4.8 + rand*(-4.4-(-4.8));
    mlp4(4,1) = -5.1 + rand*(-4.7-(-5.1));
    mlp4(5,1) = -11.5 + rand*(-11.1-(-11.5));
    mlp4(6,1) = -6.9 + rand*(-6.5-(-6.9));
    mlp4(7,1) = -13.2 + rand*(-12.8-(-13.2));
    mlp4(8,1) = -7.6 + rand*(-7.2-(-7.6));
    mlp4(9,1) = -20.4 + rand*(-20-(-20.4));
    
    mlp5(1,1) = -3.9 + rand*(-3.5-(-3.9));
    mlp5(2,1) = -7.5 + rand*(-7.1-(-7.5));
    mlp5(3,1) = -0.8 + rand*(-0.4-(-0.8));
    mlp5(4,1) = -1.7 + rand*(-1.3-(-1.7));
    mlp5(5,1) = -10 + rand*(-9.6-(-10));
    mlp5(6,1) = -3.3 + rand*(-2.9-(-3.3));
    mlp5(7,1) = -13.2 + rand*(-12.8-(-13.2));
    mlp5(8,1) = -8.7 + rand*(-8.4-(-8.7));
    mlp5(9,1) = -10.9 + rand*(-10.5-(-10.9));


    % Training loop starts here
    training_cy = 1;
    training_order = randperm(100);
    mtraining_id = training_order(training_cy);
    while((sum_squared > .001) && (done == 0))
        mtraining_input = mtraining_data{mtraining_id,1};
        p0_desired = mtraining_data{mtraining_id,2}(8);
        p1_desired = mtraining_data{mtraining_id,2}(7);
        p2_desired = mtraining_data{mtraining_id,2}(6);
        p3_desired = mtraining_data{mtraining_id,2}(5);
        p4_desired = mtraining_data{mtraining_id,2}(4);
        p5_desired = mtraining_data{mtraining_id,2}(3);
        p6_desired = mtraining_data{mtraining_id,2}(2);
        p7_desired = mtraining_data{mtraining_id,2}(1);

        p0_actual = sigmf(mtraining_input*slp0,[1 0]);
        p6_actual = sigmf(mtraining_input*slp6,[1 0]);

        a_actual = sigmf(mtraining_input*hla,[1 0]);
        a_actual = [a_actual -1];
        p1_actual = sigmf(a_actual*mlp1,[1 0]);

        b_actual = sigmf(mtraining_input*hlb,[1 0]);
        b_actual = [b_actual -1];
        p2_actual = sigmf(b_actual*mlp2,[1 0]);

        c_actual = sigmf(mtraining_input*hlc,[1 0]);
        c_actual = [c_actual -1];
        p3_actual = sigmf(c_actual*mlp3,[1 0]);

        d_actual = sigmf(mtraining_input*hld,[1 0]);
        d_actual = [d_actual -1];
        p4_actual = sigmf(d_actual*mlp4,[1 0]);

        e_actual = sigmf(mtraining_input*hle,[1 0]);
        e_actual = [e_actual -1];
        p5_actual = sigmf(e_actual*mlp5,[1 0]);

        p0_error = p0_desired - p0_actual;
        p1_error = p1_desired - p1_actual;
        p2_error = p2_desired - p2_actual;
        p3_error = p3_desired - p3_actual;
        p4_error = p4_desired - p4_actual;
        p5_error = p5_desired - p5_actual;
        p6_error = p6_desired - p6_actual;


        e_record(mtraining_id,1) = p0_error;
        e_record(mtraining_id,2) = p1_error;
        e_record(mtraining_id,3) = p2_error;
        e_record(mtraining_id,4) = p3_error;
        e_record(mtraining_id,5) = p4_error;
        e_record(mtraining_id,6) = p5_error;
        e_record(mtraining_id,7) = p6_error;

        egradp0 = p0_actual * (1-p0_actual) * p0_error;
        egradp1 = p1_actual * (1-p1_actual) * p1_error;
        egradp2 = p2_actual * (1-p2_actual) * p2_error;
        egradp3 = p3_actual * (1-p3_actual) * p3_error;
        egradp4 = p4_actual * (1-p4_actual) * p4_error;
        egradp5 = p5_actual * (1-p5_actual) * p5_error;
        egradp6 = p6_actual * (1-p6_actual) * p6_error;

        for i = 1:9
            dslp0(i,1) = egradp0*alpha*mtraining_data{mtraining_id,1}(i);
            dslp6(i,1) = egradp6*alpha*mtraining_data{mtraining_id,1}(i);
        end

        for j = 1:9
            dmlp1(j,1) = egradp1*alpha*a_actual(j);
            dmlp2(j,1) = egradp2*alpha*b_actual(j);
            dmlp3(j,1) = egradp3*alpha*c_actual(j);
            dmlp4(j,1) = egradp4*alpha*d_actual(j);
            dmlp5(j,1) = egradp5*alpha*e_actual(j);
        end

        egrada1 = a_actual(1) * (1-a_actual(1)) * (mlp1(1)*egradp1);
        egrada2 = a_actual(2) * (1-a_actual(2)) * (mlp1(2)*egradp1);
        egrada3 = a_actual(3) * (1-a_actual(3)) * (mlp1(3)*egradp1);
        egrada4 = a_actual(4) * (1-a_actual(4)) * (mlp1(4)*egradp1);
        egrada5 = a_actual(5) * (1-a_actual(5)) * (mlp1(5)*egradp1);
        egrada6 = a_actual(6) * (1-a_actual(6)) * (mlp1(6)*egradp1);
        egrada7 = a_actual(7) * (1-a_actual(7)) * (mlp1(7)*egradp1);
        egrada8 = a_actual(8) * (1-a_actual(8)) * (mlp1(8)*egradp1);

        egradb1 = b_actual(1) * (1-b_actual(1)) * (mlp2(1)*egradp2);
        egradb2 = b_actual(2) * (1-b_actual(2)) * (mlp2(2)*egradp2);
        egradb3 = b_actual(3) * (1-b_actual(3)) * (mlp2(3)*egradp2);
        egradb4 = b_actual(4) * (1-b_actual(4)) * (mlp2(4)*egradp2);
        egradb5 = b_actual(5) * (1-b_actual(5)) * (mlp2(5)*egradp2);
        egradb6 = b_actual(6) * (1-b_actual(6)) * (mlp2(6)*egradp2);
        egradb7 = b_actual(7) * (1-b_actual(7)) * (mlp2(7)*egradp2);
        egradb8 = b_actual(8) * (1-b_actual(8)) * (mlp2(8)*egradp2);

        egradc1 = c_actual(1) * (1-c_actual(1)) * (mlp3(1)*egradp3);
        egradc2 = c_actual(2) * (1-c_actual(2)) * (mlp3(2)*egradp3);
        egradc3 = c_actual(3) * (1-c_actual(3)) * (mlp3(3)*egradp3);
        egradc4 = c_actual(4) * (1-c_actual(4)) * (mlp3(4)*egradp3);
        egradc5 = c_actual(5) * (1-c_actual(5)) * (mlp3(5)*egradp3);
        egradc6 = c_actual(6) * (1-c_actual(6)) * (mlp3(6)*egradp3);
        egradc7 = c_actual(7) * (1-c_actual(7)) * (mlp3(7)*egradp3);
        egradc8 = c_actual(8) * (1-c_actual(8)) * (mlp3(8)*egradp3);

        egradd1 = d_actual(1) * (1-d_actual(1)) * (mlp4(1)*egradp4);
        egradd2 = d_actual(2) * (1-d_actual(2)) * (mlp4(2)*egradp4);
        egradd3 = d_actual(3) * (1-d_actual(3)) * (mlp4(3)*egradp4);
        egradd4 = d_actual(4) * (1-d_actual(4)) * (mlp4(4)*egradp4);
        egradd5 = d_actual(5) * (1-d_actual(5)) * (mlp4(5)*egradp4);
        egradd6 = d_actual(6) * (1-d_actual(6)) * (mlp4(6)*egradp4);
        egradd7 = d_actual(7) * (1-d_actual(7)) * (mlp4(7)*egradp4);
        egradd8 = d_actual(8) * (1-d_actual(8)) * (mlp4(8)*egradp4);

        egrade1 = e_actual(1) * (1-e_actual(1)) * (mlp5(1)*egradp5);
        egrade2 = e_actual(2) * (1-e_actual(2)) * (mlp5(2)*egradp5);
        egrade3 = e_actual(3) * (1-e_actual(3)) * (mlp5(3)*egradp5);
        egrade4 = e_actual(4) * (1-e_actual(4)) * (mlp5(4)*egradp5);
        egrade5 = e_actual(5) * (1-e_actual(5)) * (mlp5(5)*egradp5);
        egrade6 = e_actual(6) * (1-e_actual(6)) * (mlp5(6)*egradp5);
        egrade7 = e_actual(7) * (1-e_actual(7)) * (mlp5(7)*egradp5);
        egrade8 = e_actual(8) * (1-e_actual(8)) * (mlp5(8)*egradp5);

        for k = 1:9
            da1(k,1) = alpha*egrada1*mtraining_data{mtraining_id,1}(k);
            da2(k,1) = alpha*egrada2*mtraining_data{mtraining_id,1}(k);
            da3(k,1) = alpha*egrada3*mtraining_data{mtraining_id,1}(k);
            da4(k,1) = alpha*egrada4*mtraining_data{mtraining_id,1}(k);
            da5(k,1) = alpha*egrada5*mtraining_data{mtraining_id,1}(k);
            da6(k,1) = alpha*egrada6*mtraining_data{mtraining_id,1}(k);
            da7(k,1) = alpha*egrada7*mtraining_data{mtraining_id,1}(k);
            da8(k,1) = alpha*egrada8*mtraining_data{mtraining_id,1}(k);

            db1(k,1) = alpha*egradb1*mtraining_data{mtraining_id,1}(k);
            db2(k,1) = alpha*egradb2*mtraining_data{mtraining_id,1}(k);
            db3(k,1) = alpha*egradb3*mtraining_data{mtraining_id,1}(k);
            db4(k,1) = alpha*egradb4*mtraining_data{mtraining_id,1}(k);
            db5(k,1) = alpha*egradb5*mtraining_data{mtraining_id,1}(k);
            db6(k,1) = alpha*egradb6*mtraining_data{mtraining_id,1}(k);
            db7(k,1) = alpha*egradb7*mtraining_data{mtraining_id,1}(k);
            db8(k,1) = alpha*egradb8*mtraining_data{mtraining_id,1}(k);

            dc1(k,1) = alpha*egradc1*mtraining_data{mtraining_id,1}(k);
            dc2(k,1) = alpha*egradc2*mtraining_data{mtraining_id,1}(k);
            dc3(k,1) = alpha*egradc3*mtraining_data{mtraining_id,1}(k);
            dc4(k,1) = alpha*egradc4*mtraining_data{mtraining_id,1}(k);
            dc5(k,1) = alpha*egradc5*mtraining_data{mtraining_id,1}(k);
            dc6(k,1) = alpha*egradc6*mtraining_data{mtraining_id,1}(k);
            dc7(k,1) = alpha*egradc7*mtraining_data{mtraining_id,1}(k);
            dc8(k,1) = alpha*egradc8*mtraining_data{mtraining_id,1}(k);

            dd1(k,1) = alpha*egradd1*mtraining_data{mtraining_id,1}(k);
            dd2(k,1) = alpha*egradd2*mtraining_data{mtraining_id,1}(k);
            dd3(k,1) = alpha*egradd3*mtraining_data{mtraining_id,1}(k);
            dd4(k,1) = alpha*egradd4*mtraining_data{mtraining_id,1}(k);
            dd5(k,1) = alpha*egradd5*mtraining_data{mtraining_id,1}(k);
            dd6(k,1) = alpha*egradd6*mtraining_data{mtraining_id,1}(k);
            dd7(k,1) = alpha*egradd7*mtraining_data{mtraining_id,1}(k);
            dd8(k,1) = alpha*egradd8*mtraining_data{mtraining_id,1}(k);

            de1(k,1) = alpha*egrade1*mtraining_data{mtraining_id,1}(k);
            de2(k,1) = alpha*egrade2*mtraining_data{mtraining_id,1}(k);
            de3(k,1) = alpha*egrade3*mtraining_data{mtraining_id,1}(k);
            de4(k,1) = alpha*egrade4*mtraining_data{mtraining_id,1}(k);
            de5(k,1) = alpha*egrade5*mtraining_data{mtraining_id,1}(k);
            de6(k,1) = alpha*egrade6*mtraining_data{mtraining_id,1}(k);
            de7(k,1) = alpha*egrade7*mtraining_data{mtraining_id,1}(k);
            de8(k,1) = alpha*egrade8*mtraining_data{mtraining_id,1}(k);
        end

        dhla = [da1, da2, da3, da4, da5, da6, da7, da8];
        dhlb = [db1, db2, db3, db4, db5, db6, db7, db8];
        dhlc = [dc1, dc2, dc3, dc4, dc5, dc6, dc7, dc8];
        dhld = [dd1, dd2, dd3, dd4, dd5, dd6, dd7, dd8];
        dhle = [de1, de2, de3, de4, de5, de6, de7, de8];

        slp0 = slp0 + dslp0;
        slp6 = slp6 + dslp6;

        mlp1 = mlp1 + dmlp1;
        mlp2 = mlp2 + dmlp2;
        mlp3 = mlp3 + dmlp3;
        mlp4 = mlp4 + dmlp4;
        mlp5 = mlp5 + dmlp5;

        hla = hla + dhla;
        hlb = hlb + dhlb;
        hlc = hlc + dhlc;
        hld = hld + dhld;
        hle = hle + dhle;

        e_squared = (e_record).^2;
        sum_squared = sum(e_squared(:));

        epoch = epoch + 1;
        training_cy = training_cy + 1;


        if (training_cy > 100)
            training_cy = 1;
            training_order = randperm(100);
        end

       mtraining_id = training_order(training_cy);


       if (mod(epoch,20000) == 0)
            epoch
            sum_squared
       end

       if(max(e_squared(:)) > .01)
           done = 0;
       else
           done = 1;
       end
    end

    
    mult_trained = 1;
    
    binary_in1 = (de2bi(summand_1,4,'left-msb'));
    binary_in2 = (de2bi(summand_2,4,'left-msb'));
    multi_input = [binary_in1, binary_in2, -1];

    p0 = sigmf(multi_input*slp0,[1 0]);

    a = sigmf(multi_input*hla,[1 0]);
    a = [a -1];
    p1 = sigmf(a*mlp1,[1 0]);

    b = sigmf(multi_input*hlb,[1 0]);
    b = [b -1];
    p2 = sigmf(b*mlp2,[1 0]);

    c = sigmf(multi_input*hlc,[1 0]);
    c = [c -1];
    p3 = sigmf(c*mlp3,[1 0]);

    d = sigmf(multi_input*hld,[1 0]);
    d = [d -1];
    p4 = sigmf(d*mlp4,[1 0]);

    e = sigmf(multi_input*hle,[1 0]);
    e = [e -1];
    p5 = sigmf(e*mlp5,[1 0]);

    p6 = sigmf(multi_input*slp6,[1 0]);

    output = [p6,p5,p4,p3,p2,p1,p0];

    for i = 1:7
        if output(i) < .2
            output(i) = 0;
        end
        if output(i) > .9
            output(i) = 1;
        end
    end

    output
    output_dec = bi2de(output,'left-msb');
    output_dec
end

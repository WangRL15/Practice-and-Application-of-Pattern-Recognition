
 
input = readmatrix("iris_in.csv")
output_temp = readmatrix("iris_out.csv")
output = [];
 
for i =  1:150
    if(output_temp(i, 1) == 1)
        output = [output; 1, 0, 0];
    elseif(output_temp(i, 1) == 2)
        output = [output; 0, 1, 0]
    else
        output = [output; 0, 0, 1]
    end
end
 
training_in = input(1:75,:)
training_out = output(1:75,:)
test_in = input(76:150,:)
test_out = output(76:150,:)
 
Whid = rand(4, 10)
Wout = rand(10, 3)
BAhid = rand(1, 10)
BAout = rand(1, 3)
 
RMSE = []
Epochs = []
alpha = 0.1
 
for epoch = 1:100
    MSE = 0;
    for i = 1:75
        input = training_in(i,:)
        target = training_out(i,:)
        SUMhid = input*Whid + BAhid
        Ahid = logsig(SUMhid)
 
        SUMout  = Ahid*Wout + BAout;
        Aout = purelin(SUMout)
 
        DELTAout = target - Aout
        DELTAhid = DELTAout.*dpurelin(Aout) * Wout';
 
        MSE = MSE + mean((target-Aout).^2);
 
        Wout = Wout + Ahid.'*DELTAout.*dpurelin(Aout)*alpha;
        BAout = BAout + DELTAout.*dpurelin(Aout)*alpha;
        Whid = Whid + input.'*DELTAhid.*dlogsig(SUMhid, Ahid)*alpha;
        BAhid = BAhid + DELTAhid.*dlogsig(SUMhid, Ahid)*alpha;
    end
 
    rmse = sqrt(MSE/75)
    RMSE = [RMSE, rmse];
    Epochs = [Epochs, epoch];
 
end
final_output = []
accuracy = 0;
 
for i = 1:75
    input = test_in(i,:)
    target = test_out(i, :)
 
    SUMhid = input*Whid+BAhid
    Ahid = logsig(SUMhid)
 
    SUMout = Ahid*Wout+BAout;
    Aout = purelin(SUMout)
 
    [max_value, max_index] = max(Aout);
    if(max_index == output_temp(i+75))
        accuracy = accuracy + 1
    end
 
    final_output = [final_output; Aout]
end
 
plot(Epochs, RMSE)
fprintf('Accuracy: %f', accuracy/75)
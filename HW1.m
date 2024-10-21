load iris_in.csv;
load iris_out.csv;
input = iris_in;
target = iris_out;
  
  
% Initialize: Wout
outputmatrix = zeros(12, 1);
for i = 1:1:12
    for j = 1:1:1
        % random(-1, 1)
        outputmatrix(i, j) = 2*rand - 1;
    end
end
  
% Initialize: Whid
hiddenmatrix = zeros(4, 12);
for i = 1:1:4
    for j = 1:1:12
        % random(-1, 1)
        hiddenmatrix(i, j) = 2*rand - 1;
    end
end
  
RMSE = zeros(100, 1);
  
% start training
% using first 75 data
for epoch = 1:1:100
t = [];
  
for iter = 1:1:75
    % 前傳
    % SUMhid = Whid * P
    SUMhid = input(iter,:)*hiddenmatrix;
    Ahid = logsig(SUMhid);
    % SUMout = Ahid * Wout
    SUMout = Ahid * outputmatrix;
    Aout = purelin(SUMout);
  
  
    % 倒傳
    % DELTAout = (T-Aout) * dpurein(Aout)
    Dout = (target(iter) - Aout);
    dTransferOut = dpurelin(SUMout);
    error = target(iter) - Aout;
    t = [t; error.^2];
  
    % DELTAhid = DELTAout * Wout * dlogsig
    Dhid = Dout * dTransferOut * outputmatrix;
    dTransferHid = dlogsig(Ahid, logsig(SUMhid));
  
    % update weight
    % out layer
    outputmatrix = outputmatrix + 0.45 * Dout * dTransferOut * Ahid';
  
    % update weight of hidden layer
    for i = 1:1:12
        hiddenmatrix(:, i) = hiddenmatrix(:, i) + 0.45 * Dhid(i) * dTransferHid(i);
    end
  
end
  
RMSE(epoch) = sqrt(sum(t)/75);
fprintf('epoch %.0f:  RMSE = %.3f\n', epoch, sqrt(sum(t)/75));
end
  
fprintf('\nToyal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE(epoch));
  
% plot
plot(1 : epoch, RMSE(1 : epoch))
legend('Training');
xlabel('Epoch');
ylabel('RMSE');
  
  
% Last 75 data 
correct = 0;
testing_result = [];
result = []
  
for i = 76 : length(input)
    sigmaHid = input(i,:) * hiddenmatrix;
    netHid = logsig(sigmaHid);
    sigmaOut = netHid * outputmatrix;
    netOut = purelin(sigmaOut);
    testing_result = [testing_result;netOut];
        if (netOut > target(i) - 0.5) && (netOut <= target(i) + 0.5)
            correct = correct + 1;
        end
end
  
percent = (correct) / (length(input) - 75);
correct_percent = percent
  
RMSE(epoch) = sqrt(sum(t)/75);
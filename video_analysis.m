
function video_analysis( vacpath )

% Get the file name from the full file path.
% "FileName" is the variable that is containing the file name.
[~,FileName,~] = fileparts(vacpath);
disp(FileName);


% Creating folder for the video data.
% Check if the folder already exist or not.
% If there is already a folder with same name it includes 'copy' to it.
FCreate1= fullfile(FileName);
FCreate2= fullfile(sprintf('%s copy',FileName));
if (exist(FCreate1,'dir') == 0)
   mkdir(FCreate1);
   FolderPath= sprintf('%s/%s',FileName);
else
   mkdir(FCreate2);
   FolderPath= sprintf('%s/%s copy',FileName);
end


% Detach audio from the video file.
% Save audio in wav format as 'audio.wav'.
input_file = audioread(vacpath);
info_of_file = audioinfo(vacpath);
audiowrite(sprintf('%s/audio.wav',FolderPath), input_file, info_of_file.SampleRate);


% Extracting frames from the video.
% Write all the frames in the newly created folder 'Frames'.
mov = VideoReader(vacpath);
% Output folder
opFolder = fullfile(FolderPath,'Frames');
if ~exist(opFolder, 'dir')
% make directory
mkdir(opFolder);
end
% Getting the number of total frames.
numFrames = mov.NumberOfFrames;
% Counting the number of written frames.
numFramesWritten = 1;
for t = 1 : 50 : numFrames
currFrame = read(mov, t);    
opBaseFileName = sprintf('%d.jpg', numFramesWritten);
opFullFileName = fullfile(opFolder, opBaseFileName);
imwrite(currFrame, opFullFileName, 'jpg');   %saving as 'jpg' file
numFramesWritten = numFramesWritten + 1;
end

 TotalFrames= numFramesWritten - 1;
% progIndication = sprintf('Wrote %d frames to folder "%s"',TotalFrames, opFolder);
% disp(progIndication);
% disp(TotalFrames);



% Creating a folder 'VideoData'.
FCreate3= fullfile(FolderPath,'VideoData');
mkdir(FCreate3);

% Creating a specific path.
% Creating path for 'frame.csv' file.
Video_Data_Path = sprintf('%s/VideoData',FolderPath);
Frame_Data_File = sprintf('%s/frame.csv',Video_Data_Path);









ArrayNum=1;
% Starting of for loop for image analysis.
%2:2
for num = 2 : 1 : TotalFrames
    
    % Get the image path.
    ImageName = sprintf('%s/Frames/%d.jpg',FolderPath,num);
    % Read Image.
    BW = imread(ImageName);
    I = rgb2gray(BW);
    
    
disp('Image');
disp(num);


% Calculating color percentage for all images.
% take nearly 2 seconds to compute for an individual image.
% For one colored image it gives NaN as the result.
rgb_percent = squeeze(sum(sum(BW,1),2))/sum(BW(:))*100;

if (isnan(rgb_percent))
RedColorPercent(ArrayNum) = 33.34;
GreenColorPercent(ArrayNum) = 33.33;
BlueColorPercent(ArrayNum) = 33.33;
else
RedColorPercent(ArrayNum) = rgb_percent(1);
GreenColorPercent(ArrayNum) = rgb_percent(2);
BlueColorPercent(ArrayNum) = rgb_percent(3);
end    

disp('Color Percentage: Red,Green,BLue');
disp(RedColorPercent(ArrayNum));
F1 = RedColorPercent(ArrayNum);
disp(GreenColorPercent(ArrayNum));
F2 = GreenColorPercent(ArrayNum);
disp(BlueColorPercent(ArrayNum));
F3 = BlueColorPercent(ArrayNum);


% Finding number of objects.
cc = bwconncomp(BW,4);
NumberOfObjects(ArrayNum)  = cc.NumObjects;
disp('Number of objects')
disp(NumberOfObjects(ArrayNum));
F4 = NumberOfObjects(ArrayNum);


% Calculating Area of object.
AreaOfObjects(ArrayNum) = bwarea(I);
disp('Area of object');
disp(AreaOfObjects(ArrayNum));
F5 = AreaOfObjects(ArrayNum);

% Number of edges.
ne = edge(I,'canny');
numberOfBins = 256;
[r, cl, x] = size(ne);
[pixelCount, grayLevels] = imhist(ne);
NumberOfEdges(ArrayNum) = nnz(ne);
disp('Number of edges');
disp(NumberOfEdges(ArrayNum));
F6 = NumberOfEdges(ArrayNum);

% Calculating Euler numbr of the image.
EulerNumber(ArrayNum)=bweuler(I);
disp('Euler Number')
disp(EulerNumber(ArrayNum));
F7 = EulerNumber(ArrayNum);

% Image gradient magnitude & gradient direction.
[Gx, Gy] = imgradientxy(I);
[Gmag, Gdir] = imgradient(Gx, Gy);
TempGM=mean(Gmag);
TempGD=mean(Gdir);

GradientMagnitude(ArrayNum)=mean(TempGM);
GradientDirection(ArrayNum)=mean(TempGD);
disp('Gradient Magnitude & Direction');
disp(GradientMagnitude(ArrayNum));
F8 = GradientMagnitude(ArrayNum);
disp(GradientDirection(ArrayNum));
F9 = GradientDirection(ArrayNum);

% Mean Squarred error
Alter = imnoise(BW,'salt & pepper', 0.02);
MeanSquarredError(ArrayNum) = immse(Alter, BW);
disp('Mean Squarred Error');
disp(MeanSquarredError(ArrayNum));
F10 = MeanSquarredError(ArrayNum);

%SNR= Signal to Noise Ratio
[PeakSNR(ArrayNum), SNR(ArrayNum)] = psnr(Alter, BW);
disp('Peak SNR');
disp(PeakSNR(ArrayNum));
F11 = PeakSNR(ArrayNum);

if (isinf(SNR(ArrayNum)))
    SNR(ArrayNum) = 0.95;
end

disp('SNR');
disp(SNR(ArrayNum));
F12 = SNR(ArrayNum);


% Getting Co-relation Co-eficient.
disp('CorrealationCoef');
JCC = medfilt2(I);
CorrealationCoef(ArrayNum) = corr2(I,JCC);

if (isnan(CorrealationCoef(ArrayNum)))
    CorrealationCoef(ArrayNum) = -4.5;
end

disp(CorrealationCoef(ArrayNum));
F13 = CorrealationCoef(ArrayNum);

% Getting mean of matrix.
disp('MeanOfMatrix');
MeanOfMatrix(ArrayNum) = mean2(I);
disp(MeanOfMatrix(ArrayNum));
F14 = MeanOfMatrix(ArrayNum);

%Getting Standard Deviation.
disp('StandardDeviation');
StandardDeviation(ArrayNum) = std2(I);
disp(StandardDeviation(ArrayNum));
F15 = StandardDeviation(ArrayNum);

% Getting Entropy of image.
Entropy(ArrayNum) = entropy(I);
 disp('Entopy');
 disp(Entropy(ArrayNum));
 F16 = Entropy(ArrayNum);


%Weight of Image Pixel Based on Gradient.
%Take Averrage Time to Compute.
sigma = 1.5;
TempGW = mean(gradientweight(I, sigma, 'RolloffFactor', 3, 'WeightCutoff', 0.25));
ImagePixelWeihgt(ArrayNum)=mean(TempGW);
disp('Pixel Weight');
disp(ImagePixelWeihgt(ArrayNum));
F17 = ImagePixelWeihgt(ArrayNum);

%Global image threshold.
%Take only seconds to compute.
ImageThreshold(ArrayNum) = graythresh(I);
disp('Image Thresh Hold');
disp(ImageThreshold(ArrayNum));
F18 = ImageThreshold(ArrayNum);


% Getting all values in an array.
Frame_Data = [F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 F16 F17 F18];
% Write values to the csv file
dlmwrite(Frame_Data_File,Frame_Data,'-append');


ArrayNum= ArrayNum+1;
    
end


Avg_RedColorPercent= mean(RedColorPercent);
Deviation_RedColorPercent= std(RedColorPercent);
fprintf('Average of Red Color Percentage: %d\n',Avg_RedColorPercent);
fprintf('Deviation of Red Color Percentage: %d\n',Deviation_RedColorPercent);

Avg_GreenColorPercent= mean(GreenColorPercent);
Deviation_GreenColorPercent= std(GreenColorPercent);
fprintf('Average of Green Color Percentage: %d\n',Avg_GreenColorPercent);
fprintf('Deviation of Green Color Percentage: %d\n',Deviation_GreenColorPercent);

Avg_BlueColorPercent= mean(BlueColorPercent);
Deviation_BlueColorPercent= std(BlueColorPercent);
fprintf('Average of Blue Color Percentage: %d\n',Avg_BlueColorPercent);
fprintf('Deviation of Blue Color Percentage: %d\n',Deviation_BlueColorPercent);

Avg_NumberOfObjects= mean(NumberOfObjects);
Deviation_NumberOfObjects= std(NumberOfObjects);
fprintf('Average of Number of Objects: %d\n',Avg_NumberOfObjects);
fprintf('Deviation of Number of Objects: %d\n',Deviation_NumberOfObjects);

Avg_AreaOfObjects= mean(AreaOfObjects);
Deviation_AreaOfObjects= std(AreaOfObjects);
fprintf('Average of Area of Objects: %d\n',Avg_AreaOfObjects);
fprintf('Deviation of Area of Objects: %d\n',Deviation_AreaOfObjects);

Avg_NumberOfEdges= mean(NumberOfEdges);
Deviation_NumberOfEdges= std(NumberOfEdges);
fprintf('Average of Number of Edges: %d\n',Avg_NumberOfEdges);
fprintf('Deviation of Number of Edges: %d\n',Deviation_NumberOfEdges);

Avg_EulerNumber= mean(EulerNumber);
Deviation_EulerNumber= std(EulerNumber);
fprintf('Average of Euler Number: %d\n',Avg_EulerNumber);
fprintf('Deviation of Euler Number: %d\n',Deviation_EulerNumber);

Avg_GradientMagnitude= mean(GradientMagnitude);
Deviation_GradientMagnitude= std(GradientMagnitude);
fprintf('Average of Gradient Magnitude: %d\n',Avg_GradientMagnitude);
fprintf('Deviation of Gradient Magnitude: %d\n',Deviation_GradientMagnitude);

Avg_GradientDirection= mean(GradientDirection);
Deviation_GradientDirection= std(GradientDirection);
fprintf('Average of Gradient Direction: %d\n',Avg_GradientDirection);
fprintf('Deviation of Gradient Direction: %d\n',Deviation_GradientDirection);

Avg_MeanSquarredError= mean(MeanSquarredError);
Deviation_MeanSquarredError= std(MeanSquarredError);
fprintf('Average of Mean Squrred Error: %d\n',Avg_MeanSquarredError);
fprintf('Deviation of Mean Squarred Error: %d\n',Deviation_MeanSquarredError);

Avg_PeakSNR= mean(PeakSNR);
Deviation_PeakSNR= std(PeakSNR);
fprintf('Average of Peak SNR : %d\n',Avg_PeakSNR);
fprintf('Deviation of Peak SNR: %d\n',Deviation_PeakSNR);

Avg_SNR= mean(SNR);
Deviation_SNR= std(SNR);
fprintf('Average of SNR: %d\n',Avg_SNR);
fprintf('Deviation of SNR: %d\n',Deviation_SNR);

Avg_CorrealationCoef= mean(CorrealationCoef);
Deviation_CorrealationCoef= std(CorrealationCoef);
fprintf('Average of Correlation Co-efficient: %d\n',Avg_CorrealationCoef);
fprintf('Deviation of Correlation Co-efficient: %d\n',Deviation_CorrealationCoef);

Avg_MeanOfMatrix= mean(MeanOfMatrix);
Deviation_MeanOfMatrix= std(MeanOfMatrix);
fprintf('Average of Mean of Matrix: %d\n',Avg_MeanOfMatrix);
fprintf('Deviation of Mean of Matrix : %d\n',Deviation_MeanOfMatrix);

Avg_StandardDeviation= mean(StandardDeviation);
Deviation_StandardDeviation= std(StandardDeviation);
fprintf('Average of Standard Deviation: %d\n',Avg_StandardDeviation);
fprintf('Deviation of Standard Deviation: %d\n',Deviation_StandardDeviation);

Avg_Entropy= mean(Entropy);
Deviation_Entropy= std(Entropy);
fprintf('Average of Entropy: %d\n',Avg_Entropy);
fprintf('Deviation of Entropy: %d\n',Deviation_Entropy);

Avg_ImagePixelWeihgt= mean(ImagePixelWeihgt);
Deviation_ImagePixelWeihgt= std(ImagePixelWeihgt);
fprintf('Average of Image Pixel Weihgt: %d\n',Avg_ImagePixelWeihgt);
fprintf('Deviation of Image Pixel Weihgt: %d\n',Deviation_ImagePixelWeihgt);

Avg_ImageThreshold= mean(ImageThreshold);
Deviation_ImageThreshold= std(ImageThreshold);
fprintf('Average of Image Threshold: %d\n',Avg_ImageThreshold);
fprintf('Deviation of Image Threshold: %d\n',Deviation_ImageThreshold);






% Creating a folder 'Audio'.
FCreate4= fullfile(FolderPath,'AudioSignal');
mkdir(FCreate4);

% Creating a specific path.
Audio_Data_Path = sprintf('%s/AudioSignal',FolderPath);




%Audio Analysis
AudioPath = sprintf('%s/audio.wav',FolderPath);
[x, fs] = audioread(AudioPath); % read the file
x = x(:, 1);                    % get the first channel
N = length(x);                  % signal length
t = (0:N-1)/fs;


% plot the signal waveform
figure(1);
% figure(1) = figure('visible','off')
% set(figure(1), 'Visible', 'off');
plot(t, x, 'b');
xlim([0 max(t)]);
ylim([-1.1*max(abs(x)) 1.1*max(abs(x))]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
xlabel('Time, s');
ylabel('Signal amplitude');
title('The signal in the time domain');

% Save the figure in the folder and close that.
saveas(gcf,sprintf('%s/SignalWave.jpg',Audio_Data_Path));
close(figure(1));





% plot the signal spectrogram
figure(2);
% set(figure(2), 'visible','off')
spectrogram(x, 1024, 3/4*1024, [], fs, 'yaxis');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
xlabel('Time, s');
ylabel('Frequency, Hz');
title('Spectrogram of the signal');

h = colorbar;
set(h, 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel(h, 'Magnitude, dB');
% 
% spectral analysis
win = hanning(N);           % window
K = sum(win)/N;             % coherent amplification of the window
X = abs(fft(x.*win))/N;     % FFT of the windowed signal
NUP = ceil((N+1)/2);        % calculate the number of unique points
X = X(1:NUP);               % FFT is symmetric, throw away second half 
if rem(N, 2)                % odd nfft excludes Nyquist point
  X(2:end) = X(2:end)*2;
else                        % even nfft includes Nyquist point
  X(2:end-1) = X(2:end-1)*2;
end
f = (0:NUP-1)*fs/N ;        % frequency vector
X = 20*log10(X);            % spectrum magnitude

% Save the figure in the folder and close that.
saveas(figure(2),sprintf('%s/Spectrogram.jpg',Audio_Data_Path));
close(figure(2));



% plot the signal spectrum
figure(3);
semilogx(f, X, 'r');
xlim([0 max(f)]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
title('Amplitude spectrum of the signal');
xlabel('Frequency, Hz');
ylabel('Magnitude, dB');

% Save the figure in the folder and close that.
saveas(figure(3),sprintf('%s/Spectrum.jpg',Audio_Data_Path));
close(figure(3));



% plot the signal histogram
figure(4);
histfit(x);
xlim([-1.1*max(abs(x)) 1.1*max(abs(x))]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
xlabel('Signal amplitude');
ylabel('Number of samples');
title('Probability distribution of the signal');
legend('probability distribution of the signal',...
       'standard normal distribution');

% autocorrelation function estimation
[Rx, lags] = xcorr(x, 'coeff');
d = lags/fs;

% Save the figure in the folder and close that.
saveas(figure(4),sprintf('%s/Histogram.jpg',Audio_Data_Path));
close(figure(4));


% plot the signal autocorrelation function
figure(5);
plot(d, Rx, 'r');
grid on;
xlim([-max(d) max(d)]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
xlabel('Delay, s');
ylabel('Autocorrelation coefficient');
title('Autocorrelation of the signal');
line([-max(abs(d)) max(abs(d))], [0.05 0.05],...
     'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');

% Save the figure in the folder and close that.
saveas(figure(5),sprintf('%s/Autocorrelation.jpg',Audio_Data_Path));
close(figure(5));







%compute and display the minimum and maximum values
maxval = max(x);
minval = min(x);
disp(['Max value = ' num2str(maxval)])
disp(['Min value = ' num2str(minval)])
 
% compute and display the the DC and RMS values
meanval = mean(x);
rmsval = std(x);
disp(['Mean value = ' num2str(meanval)])
disp(['RMS value = ' num2str(rmsval)])

% compute and display the dynamic range
DynamicRange = 20*log10(maxval/min(abs(nonzeros(x))));
disp(['Dynamic range = ' num2str(DynamicRange) ' dB'])

% compute and display the crest factor
CrestFactor = 20*log10(maxval/rmsval);
disp(['Crest factor = ' num2str(CrestFactor) ' dB'])

% compute and display the autocorrelation time
%  [Rx, lags] = xcorr(x, 'coeff');
%  d = lags/fs;

 
ind = find(Rx>0.05, 1, 'last');
AutoCorrelation = (ind-N)/fs;
disp(['Autocorrelation time = ' num2str(AutoCorrelation) ' s'])



% Write data to CSV file for external use.
% Making a CSV(comma separated value) file to save data.
Audio_Data_File = sprintf('%s/audio.csv',Video_Data_Path);

%Writting data to the csv file.
dlmwrite(Audio_Data_File,maxval,'-append');
dlmwrite(Audio_Data_File,minval,'-append');
dlmwrite(Audio_Data_File,meanval,'-append');
dlmwrite(Audio_Data_File,rmsval,'-append');
dlmwrite(Audio_Data_File,DynamicRange,'-append');
dlmwrite(Audio_Data_File,CrestFactor,'-append');
dlmwrite(Audio_Data_File,AutoCorrelation,'-append');




% Write data to CSV file for external use.
% Making a CSV(comma separated value) file to save data.
Video_Data_File = sprintf('%s/data.csv',Video_Data_Path);

%Writting data to the csv file.
dlmwrite(Video_Data_File,Avg_RedColorPercent,'-append');
dlmwrite(Video_Data_File,Deviation_RedColorPercent,'-append');
dlmwrite(Video_Data_File,Avg_GreenColorPercent,'-append');
dlmwrite(Video_Data_File,Deviation_GreenColorPercent,'-append');
dlmwrite(Video_Data_File,Avg_BlueColorPercent,'-append');

dlmwrite(Video_Data_File,Deviation_BlueColorPercent,'-append');
dlmwrite(Video_Data_File,Avg_NumberOfObjects,'-append');
dlmwrite(Video_Data_File,Deviation_NumberOfObjects,'-append');
dlmwrite(Video_Data_File,Avg_AreaOfObjects,'-append');
dlmwrite(Video_Data_File,Deviation_AreaOfObjects,'-append');

dlmwrite(Video_Data_File,Avg_NumberOfEdges,'-append');
dlmwrite(Video_Data_File,Deviation_NumberOfEdges,'-append');
dlmwrite(Video_Data_File,Avg_EulerNumber,'-append');
dlmwrite(Video_Data_File,Deviation_EulerNumber,'-append');
dlmwrite(Video_Data_File,Avg_GradientMagnitude,'-append');

dlmwrite(Video_Data_File,Deviation_GradientMagnitude,'-append');
dlmwrite(Video_Data_File,Avg_GradientDirection,'-append');
dlmwrite(Video_Data_File,Deviation_GradientDirection,'-append');
dlmwrite(Video_Data_File,Avg_MeanSquarredError,'-append');
dlmwrite(Video_Data_File,Deviation_MeanSquarredError,'-append');

dlmwrite(Video_Data_File,Avg_PeakSNR,'-append');
dlmwrite(Video_Data_File,Deviation_PeakSNR,'-append');
dlmwrite(Video_Data_File,Avg_SNR,'-append');
dlmwrite(Video_Data_File,Deviation_SNR,'-append');
dlmwrite(Video_Data_File,Avg_CorrealationCoef,'-append');

dlmwrite(Video_Data_File,Deviation_CorrealationCoef,'-append');
dlmwrite(Video_Data_File,Avg_MeanOfMatrix,'-append');
dlmwrite(Video_Data_File,Deviation_MeanOfMatrix,'-append');
dlmwrite(Video_Data_File,Avg_StandardDeviation,'-append');
dlmwrite(Video_Data_File,Deviation_StandardDeviation,'-append');

dlmwrite(Video_Data_File,Avg_Entropy,'-append');
dlmwrite(Video_Data_File,Deviation_Entropy,'-append');
dlmwrite(Video_Data_File,Avg_ImagePixelWeihgt,'-append');
dlmwrite(Video_Data_File,Deviation_ImagePixelWeihgt,'-append');
dlmwrite(Video_Data_File,Avg_ImageThreshold,'-append');
dlmwrite(Video_Data_File,Deviation_ImageThreshold,'-append');





end

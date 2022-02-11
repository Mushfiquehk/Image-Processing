%Authors: Armin Rashvand & Mushfique Khan
%Image Processing ---- Project 3
%Identifying Ranks and Suits for Playing Cards
clear
close all

% x = imread('KH.jpg'); C:\Users\user\Dropbox\cards\

cam = webcam;
preview(cam);
x = snapshot(cam);

%Code to detect type of image entered. RGB or Grayscale is expected
[~, ~, numberOfColorChannels] = size(x);
if numberOfColorChannels > 1
    % It's a true color RGB image.  We need to convert to gray scale.
    Igray = rgb2gray(x);
else
    % It's already gray scale.  No need to convert.
    Igray = x;
end

% Code to align
xB = double(imbinarize(Igray));
sigma = 4;
xBB=imgaussfilt(xB,sigma);
[~, Gdir] = imgradient(xBB);
Gdir(Gdir<0) = Gdir(Gdir<0)+180;
[N, ~] = histcounts(Gdir, 180);
N(1) = 0;
[~, xA] = max(N);
rot_angle = 180-xA;
x_rot = imrotate(x,rot_angle,'crop');

% Code to crop
beta = 0.65; % found by trial and error, shows the card only at this level
x_rotB = double(im2bw(x_rot, beta));
x_rotBB=imgaussfilt(x_rotB,sigma);
measurements = regionprops(x_rotBB, 'BoundingBox');
x_crop = imcrop(x_rot, measurements(1).BoundingBox);
Z=imshow(x_crop);
%Rank and Suit Detection
I=rgb2gray(x_crop);
% choose a conversion factor to binary scaled image
alpha = .5;
% convert the RGB image into a binary scaled image
BW = imbinarize(I,alpha);
BW = ~BW; % invert logic (0 is 1 and 1 is 0)
BW_filled=imfill(BW,'holes');
BW_final = bwareaopen(BW_filled, 6000);
props = regionprops(BW_final);
sortedAreas = sort([props.Area], 'descend');
boundaries = bwboundaries(BW_final);
boundaries2 = bwboundaries(BW_filled);
% Now let us center the boundaries of the image by subtracting the mean of
% the x and y values
[~, numberOfShapes] = bwlabel(BW_final);
if numberOfShapes>9
    for k = 1:length(boundaries) % for each boundary
        %center the x values that are found in the first column of boundaries
        boundaries{k}(:,1) = mean(boundaries{k}(:,1)) - boundaries{k}(:,1);
        %center the y values that are found in the second column of boundaries
        boundaries{k}(:,2) = mean(boundaries{k}(:,2)) - boundaries{k}(:,2);
    end
    [~,bb] = findpeaks(boundaries{k}(:,2)); % find the peaks of the upper boundary
    if(isempty(bb) || length(bb) ==1) % the diamonds and hearts have one peak
        upperslope = boundaries{k}(1:end/2,2);
        slope_mean = mean(diff(double(upperslope)));
        if(slope_mean > -0.7) % the hearts has a smaller slope than the diamonds
            disp('This is a Heart') % this is a hearts
        else
            disp('This is a Diamond') % this is a diamonds
        end
    end
    if(length(bb) == 2 || length(bb) == 3)
        disp('This is a Spade') % This is a spades
    end
    if(length(bb) >= 4)
        disp('This is a Club') % This is a clubs
    end
else
    for j = 1:length(boundaries2) % for each boundary
        %center the x values that are found in the first column of boundaries
        boundaries2{j}(:,1) = boundaries2{j}(:,1) - mean(boundaries2{j}(:,1));
        %center the y values that are found in the second column of boundaries
        boundaries2{j}(:,2) = mean(boundaries2{j}(:,2)) - boundaries2{j}(:,2);
    end
    [~,bb2] = findpeaks(boundaries2{j}(:,2)); % find the peaks of the upper boundary
    if(isempty(bb2) || length(bb2) == 1 ) % the diamonds and hearts have one peak
         
        if(length(bb2)==1)
            if(bb2<194 || bb2>213) % the hearts has a smaller slope than the diamonds
                disp('This is a Heart') % this is a hearts
            else
                disp('This is a Diamond') % this is a diamonds
            end
            
        end
        if(isempty(bb2))
            disp('This is a Diamond')
        end
    end
    if(length(bb2) == 2)
        disp('This is a Spade') % This is a spades
    end
    if(length(bb2) >= 3)
        disp('This is a Club') % This is a clubs
    end
end

% the number of suit objects need to be counted
cardNumber = numberOfShapes-8;
midArea = 1500000; % area of largest object (picture)
QArea = 22000;
KArea = 15000;
if(cardNumber <= 1)
    if(sortedAreas(1)>midArea)
        if(sortedAreas(end-3)>QArea)
            fprintf('This is a Queen\n');
        elseif((sortedAreas(end-3)>KArea) && (sortedAreas(end-2)>KArea))
            fprintf('This is a King\n');
        else
            fprintf('This is a Jack\n');
        end
    else
        fprintf('This is an Ace\n');
    end
else
    fprintf(['This is a ',num2str(min(cardNumber,10)),' card \n']); % this is the card number
end

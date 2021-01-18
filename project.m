clear all;
close all;

%location of the testing image and storing in ds 
outputFolder = fullfile('D:', 'AI', 'matlab', 'segmentation', 'tempdir', 'CamVid');
imgDir1 = fullfile(outputFolder,'test','images', 'img1');
imds1 = imageDatastore(imgDir1);
%////////////////////////////////////////////////////////////////////////////

%location of the images and storing to the data store imds 
outputFolder = fullfile('D:', 'AI', 'matlab', 'segmentation', 'tempdir', 'CamVid');
imgDir = fullfile(outputFolder,'images','source');
imds = imageDatastore(imgDir);

%to test if images loaded correctly
pic_num = 204;
I_raw = readimage(imds1,1);
I = histeq(I_raw);
%uncomment this to see
% %seeImage(I)
% 
% %7, 9, 10, 11 are dangers
classes = [
    "Sky"    %1
    "apartments"  %2
    "Pole"   %3
    "Road"    %4
    "Pavement" %5
    "nature"   %6
    "Warnings"   %7
    "Fence"    %8
    "Car"   %9
    "people"   %10
    "riders"     %11
]
% 
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder, 'labels');
pxds = pixelLabelDatastore(labelDir, classes, labelIDs)


cmap = camvidColorMap;
% 
tbl = countEachLabel(pxds);
% 
imageFolder = fullfile(outputFolder, 'imagesResized', filesep);
imds = resizeCamVidImages(imds,imageFolder);
% 
labelFolder = fullfile(outputFolder, 'labelsResized', filesep);
pxds = resizeCamVidPixelLabels(pxds, labelFolder)
% 
[imdsTrain, imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData(imds,pxds);
% 
imageSize = [360 480 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize, numClasses, 'vgg16');
% %uncomment to see the layers 
% %seeLayers(lgraph);
% 
% %classweights on pixelclassification must be adjusted depending on pixel counts
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq
pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights)
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax','labels');
% 
% 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64);

augmenter = imageDataAugmenter('RandXReflection', true,'RandXTranslation', [-10, 10],...
    'RandYTranslation', [-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
% 
Training = false;
% 
if Training
    [net,info] = trainNetwork(pximds, lgraph, options);
    save('PreTrainedCnn.mat','net','info', 'options');
    disp('NN trained');
else
    data = load('PreTrainedCnn.mat')
    net = data.net;    
end
checkDanger(tbl,classes)
I = readimage(imds1, 1);
C = semanticseg(I, net);

I2 = readimage(imds, 204);
C2 = semanticseg(I2, net);
 
overlay(I,C,I2,C2,cmap,classes);
%pixelCountByClass(tbl,classes)
% 
function checkDanger(tbl,classes)
   display("danger detected")
    freqObjects = tbl.PixelCount/sum(tbl.PixelCount);
    if (freqObjects(7) > 0.05) 
       display("A lot of Warning sign detected")
    end
    if (freqObjects(9) > 0.05) 
       display("A lot of Vehicles detected")
    end
    if (freqObjects(10) > 0.05) 
       display("A lot of people detected")
    end
    if (freqObjects(11) > 0.05) 
       display("A lot of riders detected detected")   
     
    end
    freqObjects(1)
end
% 
function seeLayers(lgraph)
     fig1 = figure('Position', [100,100,1000,1100]);
     subplot(1,2,1);
     plot(lgraph); 
     axis off
     axis tight
     title('Complete Layer')
 
     subplot(1,2,2)
     plot(lgraph);
     xlim([2.862, 3.200])
     ylim([-0.9 10.9])
     axis off
     title('Last 9 layers Graph')  
end
% 
function seeImage(img)
    figure;
    imshow(img);
    title('image');
end
% 
function overlay(I, C,I2,C2,cmap,classes)
   imagedata1= labeloverlay(I,C,'ColorMap',cmap);
   imagedata2 = labeloverlay(I2,C2,'ColorMap',cmap);
   subplot(1,2,1);
   imshow(imagedata1);
   subplot(1,2,2);
   imshow(imagedata2);
   pixelLabelColorbar(cmap,classes);
end
 
function pixelCountByClass(tbl,classes)
    frequency = tbl.PixelCount/sum(tbl.PixelCount);
    bar(1:numel(classes), frequency);
    xticks(1:numel(classes))
    xticklabels(tbl.Name)
    xtickangle(45)
    ylabel('Frequency')
end

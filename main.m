clear all;
close all;

outputFolder = fullfile('D:', 'AI', 'matlab', 'segmentation', 'tempdir', 'CamVid');
imgDir = fullfile(outputFolder, 'images', '701_StillsRaw_full');

imds = imageDatastore(imgDir);

pic_num = 30;
I_raw = readimage(imds, pic_num);
I = histeq(I_raw);
% seeImage(I)

classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
]

labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder, 'labels');
pxds = pixelLabelDatastore(labelDir, classes, labelIDs)

C = readimage(pxds, 1);
cmap = camvidColorMap;

tbl = countEachLabel(pxds)
 
%resizing must be done for vgg16
imageFolder = fullfile(outputFolder, 'labelsResized', filesep);
imds = resizeCamVidImages(imds, imageFolder);

labelFolder = fullfile(outputFolder, 'labelsResized', filesep);
pxds = resizeCamVidPixelLabels(pxds, labelFolder)

[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds, pxds);

imageSize = [360 240 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize, numClasses, 'vgg16');

%uncomment to see the layers of segmentation
%seeLayers(lgraph);

%classweights on pixelclassification must be adjusted depending on pixel counts

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount
classWeights = median(imageFreq) ./imageFreq
pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'classWeights', classWeights)
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax', 'labels');

options = trainingOptions('sgdm', 'InitialLearnRate', 1e-3, 'MaxEpochs', 100, 'MiniBatchSize', 64)

augmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', [-10 10],...
'RandYTranslation', [-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter)

doTraining = false;

if doTraining
   [net, info] = trainNetwork(pximds, lgraph, options);
   save('PreTrainedCnn.mat', 'net', 'info', 'options')
   disp('NN trained');
else
    data = load('PreTrainedCnn.mat')
    net = data.net;
end

I = read(imdsTest);
C = semanticseg(I, net);

overlay(I, C,cmap, classes);


function seeImage(img)
    figure;
    imshow(img);
    title('image')
end

function seeLayers(lgraph)
    fig1 = figure('Position', [100, 100,1000, 1100]);
    subplot(1,2,1);
    plot(lgraph);
    axis off
    axis tight
    title('Complete Layer')
    
    subplot(1,2,2)
    plot(lgraph)
    xlim([2.862, 3.200])
    ylim([-0.9 10.9])
    axis off
    title('Last 9 layers Graph')
end

function overlay(I, C, cmap, classes)
    B = labeloverlay(I,C,'ColorMap', cmap);
    imshow(B);
    pixelLabelColorbar(cmap, classes)
end






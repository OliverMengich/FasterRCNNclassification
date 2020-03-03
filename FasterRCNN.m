
net = resnet50();
lgraph = layerGraph(net);

layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };

lgraph = removeLayers(lgraph, layersToRemove);
numClasses=1;
numClassesPlusBackground = numClasses + 1;

% Define new classfication layers
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new layers
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');

numOutputs = 4* numClasses;

boxRegressionLayers = [
    fullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')
    rcnnBoxRegressionLayer('Name','rcnnBoxDeltas')
    ];
lgraph = addLayers(lgraph, boxRegressionLayers);

lgraph = connectLayers(lgraph,'avg_pool','rcnnBoxFC');

% Select a feature extraction layer.
featureExtractionLayer = 'activation_40_relu';


% Disconnect the layers attached to the selected feature extraction layer.
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch2a');
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch1');

% Add ROI max pooling layer.
outputSize = [14 14];
roiPool = roiMaxPooling2dLayer(outputSize,'Name','roiPool');
lgraph = addLayers(lgraph, roiPool);

% Connect feature extraction layer to ROI max pooling layer.
lgraph = connectLayers(lgraph, featureExtractionLayer,'roiPool/in');

% Connect the output of ROI max pool to the disconnected layers from above.
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch2a');
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch1');

allBoxes =vertcat(CarData.car{:});

aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);

numAnchors = 1;

% Cluster using K-Medoids.
[clusterAssignments, anchorBoxes, sumd] = kmedoids(allBoxes(:,3:4),numAnchors,'Distance',@iouDistanceMetric);


% Create the region proposal layer.
proposalLayer = regionProposalLayer(anchorBoxes,'Name','regionProposal');

lgraph = addLayers(lgraph, proposalLayer);

% Number of anchor boxes.
numAnchors = size(anchorBoxes,1);

% Number of feature maps in coming out of the feature extraction layer. 
numFilters = 1024;

rpnLayers = [
    convolution2dLayer(3, numFilters,'padding',[1 1],'Name','rpnConv3x3')
    reluLayer('Name','rpnRelu')
    ];

lgraph = addLayers(lgraph, rpnLayers);

% Connect to RPN to feature extraction layer.
lgraph = connectLayers(lgraph, featureExtractionLayer, 'rpnConv3x3');

rpnClsLayers = [
    convolution2dLayer(1, numAnchors*2,'Name', 'rpnConv1x1ClsScores')
    rpnSoftmaxLayer('Name', 'rpnSoftmax')
    rpnClassificationLayer('Name','rpnClassification')
    ];
lgraph = addLayers(lgraph, rpnClsLayers);

% Connect the classification layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1ClsScores');

rpnRegLayers = [
    convolution2dLayer(1, numAnchors*4, 'Name', 'rpnConv1x1BoxDeltas')
    rcnnBoxRegressionLayer('Name', 'rpnBoxDeltas');
    ];

lgraph = addLayers(lgraph, rpnRegLayers);

% Connect the regression layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1BoxDeltas');

lgraph = connectLayers(lgraph, 'rpnConv1x1ClsScores', 'regionProposal/scores');
lgraph = connectLayers(lgraph, 'rpnConv1x1BoxDeltas', 'regionProposal/boxDeltas');

% Connect region proposal layer to roi pooling.
lgraph = connectLayers(lgraph, 'regionProposal', 'roiPool/roi');

layers = lgraph.Layers;

%%
options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 3, ...
      'VerboseFrequency', 50, ...
      'CheckpointPath', tempdir);
  
detector = trainFasterRCNNObjectDetector(CarData,lgraph, options);
%%

imageun = imread('image_0119.jpg'); imageun = imresize(imageun,[500,500]);
[bbox, score, label] = detect(detector,imageun);
carimage =insertObjectAnnotation(imageun,'rectangle',bbox(1,:),score);
imshow(carimage)

%%
videoreader = vision.VideoFileReader('cad1.mp4');
player = vision.DeployableVideoPlayer();
while  ~isDone(videoreader)
    frame = step(videoreader);
    %create foreground mask 
    [bbox,scores] = detect(detector,frame);
    if ~isempty(bbox)
       J = insertObjectAnnotation(frame,'rectangle',bbox,scores); 
       step(player,J)
    else
        step(player,frame)
       
    end
end
%%
function dist = iouDistanceMetric(boxWidthHeight,allBoxWidthHeight)
% Return the IoU distance metric. The bboxOverlapRatio function
% is used to produce the IoU scores. The output distance is equal
% to 1 - IoU.

% Add x and y coordinates to box widths and heights so that
% bboxOverlapRatio can be used to compute IoU.
boxWidthHeight = prefixXYCoordinates(boxWidthHeight);
allBoxWidthHeight = prefixXYCoordinates(allBoxWidthHeight);

% Compute IoU distance metric.
dist = 1 - bboxOverlapRatio(allBoxWidthHeight, boxWidthHeight);
end

function boxWidthHeight = prefixXYCoordinates(boxWidthHeight)
% Add x and y coordinates to boxes.
n = size(boxWidthHeight,1);
boxWidthHeight = [ones(n,2) boxWidthHeight];
end
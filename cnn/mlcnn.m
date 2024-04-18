classdef mlcnn


properties
	class = 'mlcnn';
	nLayers;				% # OF UNIT LAYERS
	layers ;				% LAYER STRUCTS

	netOut = [];			% CURRENT OUTPUT OF THE NETWORK
	netError;				% CURRENT NETWORK OUTPUT ERROR

	costFun = 'mse';		% COST FUNCTION
	J = [];					% CURRENT COST FUNCTION VALUE

	nEpoch = 10;			% TOTAL NUMBER OF TRAINING EPOCHS
	epoch = 1;				% CURRENT EPOCH

	batchSize = 20;			% SIZE OF MINIBATCHES
	trainBatches = [];		% BATCH INDICES
	xValBatches = [];		% XVAL. DATA INDICES

	trainCost = [];			% TRAINING ERROR HISTORY (ALL DATA)

	stopEarly = 5;			% # OF EARLY STOP EPOCHS (XVAL ERROR INCREASES)
	stopEarlyCnt = 0;		% EARLY STOPPING CRITERION COUNTER
	bestNet = [];
	bestTrainCost = Inf		% TRACK BEST NETWORK ERROR% STORAGE FOR BEST NETWORK

	denoise = 0;			% PROPORTION OF VISIBLE UNIT DROPOUT (SIGMOID ONLY)
	dropout = 'not used';	% (POTENTIAL) PROP. OF HID. UNIT DROPOUT (SIGMOID ONLY)

	wPenalty = 0;			% WEIGHT DECAY TERM
	beginWeightDecay=1;
	momentum ='not used';	%0.1;% (POTENTIAL) MOMENTUM
	

	saveEvery = 1;	% SAVE PROGRESS EVERY # OF EPOCHS
	saveDir='D:\matlabyar\medal-master\save';
	visFun;					% VISUALIZATION FUNCTION HANDLE
	trainTime = Inf;		% TRAINING DURATION
	verbose = 500;			% DISPLAY THIS # OF WEIGHT UPDATES
end

methods
	function self = mlcnn(arch)
		self = self.init(arch);
	end

	function print(self)
		properties(self)
		methods(self)
	end

	function self = train(self, data, targets)
		

		self = self.makeBatches(data);
		self.trainCost = zeros(self.nEpoch,1);
		nBatches = numel(self.trainBatches);
		tic; cnt = 1;

		% MAIN
	    while 1
			if self.verbose, self.printProgress('epoch'); end
			batchCost = zeros(nBatches,1);
			wPenalty = self.wPenalty;
			
			if self.epoch >= self.beginWeightDecay
				self.wPenalty = wPenalty;
			else
				self.wPenalty = 0;
			end

			for iB = 1:nBatches
				% GET BATCH DATA
				batchIdx = self.trainBatches{iB};
				netInput = data(:,:,:,batchIdx);
				netTargets = (targets(:,batchIdx));

				% ADD NOISE TO INPUT (DENOISING CAE)
				if self.denoise > 0
				    netInput = netInput.*(rand(size(netInput))>self.denoise);
				end

				% BACKPROP MAIN
				self = self.fProp(netInput, netTargets);
				self = self.bProp;
				self = self.updateParams;

				% ASSESS BATCH COST
				batchCost(iB) = self.J;
				cnt = cnt + 1;
            end

            self=self.assessNet();
	        % AVERAGE COST OVER ALL TRAINING POINTS
			self.trainCost(self.epoch) = mean(batchCost);

			% SAVE BEST NETWORK
			if ~mod(self.epoch,self.saveEvery) & ~isempty(self.saveDir)
				self.save; 
            end

            % EARLY STOPPING
			if (self.epoch >= self.nEpoch) || ...
				(self.stopEarlyCnt >= self.stopEarly),
				self.trainCost = self.trainCost(1:self.epoch);
				break;
            end
            
			% DISPLAY
			if self.verbose
				self.printProgress('trainCost');
				if ~mod(cnt,self.verbose);
					self.visLearning;
				end
			end
			self.epoch = self.epoch + 1;
	    end
		self.trainTime = toc;

	
	end

	function self = fProp(self,netInput, targets)
		if notDefined('targets'), targets = []; end
		nObs = size(netInput, 1);		
		self.layers{1}.fm = netInput;

		for lL = 2:self.nLayers
			switch self.layers{lL}.type
			case 'conv'
			% LOOP OVER LAYER FEATURE MAPS
			for jM = 1:self.layers{lL}.nFM
				[nInY,nInX,nInFM,nObs] = size(self.layers{lL-1}.fm);
				featMap = zeros([self.layers{lL}.fmSize,1,nObs]);
				for iM = 1:self.layers{lL-1}.nFM
					featMap = featMap + convn(self.layers{lL-1}.fm(:,:,iM,:), ...
					self.layers{lL}.filter(:,:,iM,jM),'valid');
                end	
				if any(isnan(self.layers{lL}.b(jM))), keyboard, end
				% ADD LAYER BIAS
				featMap = featMap + self.layers{lL}.b(jM);
				% COMPLETE FEATURE MAP
				self.layers{lL}.fm(:,:,jM,:) = self.calcAct(featMap,self.layers{lL}.actFun);
				
			end

			case 'subsample'
				stride = self.layers{lL}.stride;
				% DOWNSAMPLE THE FEATURE MAPS FROM LAYER (l-1)
				for jM = 1:self.layers{lL-1}.nFM
					layerIn = self.layers{lL-1}.fm(:,:,jM,:);
					self.layers{lL}.fm(:,:,jM,:) = self.DOWN(layerIn,stride);
                end
            case 'hidden'
                self.layers{lL}.fm=zeros(0);
                w=self.layers{lL}.W;
                b=self.layers{lL}.b;
                w=reshape(w,size(w,1),size(w,2)*size(w,3)*size(w,4));
                if strcmp(self.layers{lL-1}.type,'hidden')
                    hnet=w*self.layers{lL-1}.fm;
                else
                    nInp=size(self.layers{lL-1}.fm);
                    hnet=w*reshape(self.layers{lL-1}.fm,[nInp(1)*nInp(2)*nInp(3),nInp(4)]);
                end
                hnet=bsxfun(@plus,hnet,b);
                self.layers{lL}.fm=self.calcAct(hnet,self.layers{lL}.actFun);
			case 'output'
				% UNPACK OUTPUT FEATURES & CALCULATE NETWORK OUTPUT
				self = self.calcOutput;
			case 'rect'
			case 'lcn'
			case 'pool'
			end
		end
		% COST FUNCTION & OUTPUT ERROR SIGNAL
		if ~isempty(targets)
			[self.J, self.netError] = self.cost(targets,self.costFun);
		end
		if nargout > 1
			out = self.layers{end}.act;
		end
	end

	function self = calcOutput(self);
		[nY,nObs,nM,nX]= size(self.layers{end-1}.fm);
		
		% # OF ENTRIES IN EACH FEATURE MAP
		nMap = prod([nY,nX]);

		% INITIALIZE OUTPUT FEATURES
		self.layers{end}.features = zeros(nMap*nM,nObs);

		% UNPACK MAPS INTO A MATRIX FOR CALCULATING OUTPUT
        
		self.layers{end}.features = self.layers{end-1}.fm;	
		% CALC NET OUTPUTS
		preAct = bsxfun(@plus,self.layers{end}.W* ...
						self.layers{end}.features, ...
		                self.layers{end}.b);
        self.netOut=zeros(0);
		self.netOut = self.calcAct(preAct,self.layers{end}.actFun);
		
		if any(isnan(self.netOut)), keyboard; end
	end

	function [J, dJ] = cost(self,targets,costFun)

		netOut = self.netOut;
	
		[nTargets,nObs] = size(netOut);
		switch costFun
		case 'mse' % REGRESSION
			delta = targets - netOut;
			J = 0.5*sum(sum(delta.^2))/nObs;
			dJ = -delta;
			
		case 'xent' % BINARY CLASSIFICATION
			J = -sum(sum(targets.*log(netOut) + (1-targets).*log(1-netOut)))/nObs;
			dJ = (netOut - targets)./(netOut.*(1-netOut));

		case 'mcxent' % MULTI-CLASS CLASSIFICATION (UNDER DEVO)
			class = softMax(netOut);
			J = -sum(sum(targets.*log(class)))/nObs;
			dJ = sum(labels - targets);

		case {'class','classerr'}  % CLASSIFICATION ERROR (WINNER TAKE ALL)
		
			[~, class] = max(netOut,[],1);
			[~, t] = max(targets,[],1);
			J = sum((class ~= t))/nObs;
			dJ = 'no gradient';
		case {'correlation','cc'}
			J = corr2(netOut,targets);
			dJ = 'no gradient';
		end
		if any(isnan(self.J)), keyboard, end
	end

	function self = bProp(self)	
		dAct = self.calcActDeriv(self.netOut,self.layers{end}.actFun);
		outES = self.netError.*dAct;
		es = self.layers{end}.W'*outES;
        
		% REPACK ERROR SIGNAL INTO 2-D FEATURE MAP REPRESENTATION
		[nY,nX,nM,nObs] = size(self.layers{end-1}.fm);
		self.layers{end-1}.es = zeros([nY,nX,nM,nObs]);
        dAct = self.calcActDeriv(self.layers{end-1}.fm,self.layers{end-1}.actFun);
		self.layers{end-1}.es=es.*dAct;

		% BACKPROPATE ERROR SIGNAL
		for lL = self.nLayers-2:-1:2
			switch self.layers{lL}.type
            case 'hidden'
                propES=self.layers{lL+1}.W'*self.layers{lL+1}.es;
                dAct = self.calcActDeriv(self.layers{lL}.fm,self.layers{lL}.actFun);
                self.layers{lL}.es=propES.*dAct;
			case 'conv'
				stride = self.layers{lL + 1}.stride;
				mapSz = size(self.layers{lL}.fm);
				self.layers{lL}.es = zeros(mapSz);

				for jM = 1:self.layers{lL}.nFM
					switch self.layers{lL+1}.type
					case 'subsample'
						% UPSAMPLE ES FROM ABOVE SUBSAMPLE LAYER
						propES = self.UP(self.layers{lL+1}.es(:,:,jM,:), ...
						                 [stride(1), stride(2),1,1])/prod(stride);
					case 'rect'
					case 'lcn'
					end

                   % DERIVATIVE OF ACTIVATION FUNCTION
                   dAct = self.calcActDeriv(self.layers{lL}.fm(:,:,jM,:), ...
											self.layers{lL}.actFun);
                   if (size(propES,1)<size(dAct,1))                     
                        propES(end+1,:,:,:)=0;
                   end
                   % CALCULATE LAYER ERROR SIGNAL
					self.layers{lL}.es(:,:,jM,:) = propES.*dAct;
				end
			case 'rect'
			case 'lcn'
			case 'pool'

			case 'subsample'
				[nY,nX,nM,nObs] = size(self.layers{lL}.fm);
                propES = zeros(nY,nX,1,nObs);
				self.layers{lL}.es = zeros([nY,nX,nM,nObs]);
                if strcmp(self.layers{lL+1}.type,'hidden')
                    es = self.layers{lL+1}.es;
					filt = self.layers{lL+1}.W(:,:,:,:);
                    filt=reshape(filt,size(filt,1),size(filt,2)*size(filt,3)*size(filt,4));
                    propES=es'*filt;
                    propES=reshape(propES,nY,nX,nM,nObs);
					self.layers{lL}.es = propES;
					if any(isnan(self.layers{lL}.es(:))), keyboard; end
					if any(isinf(self.layers{lL}.es(:))), keyboard; end
                else
				for jM = 1:self.layers{lL}.nFM
					% FORM FEATURE MAP ERROR SIGNAL
					propES = zeros(nY,nX,1,nObs);
					for kM = 1:self.layers{lL+1}.nFM
						rotFilt = self.ROT(self.layers{lL+1}.filter(:,:,jM,kM));
						es = self.layers{lL+1}.es(:,:,kM,:);
						propES = propES + convn(es,rotFilt,'full');
					end
					self.layers{lL}.es(:,:,jM,:) = propES;
					if any(isnan(self.layers{lL}.es(:))), keyboard; end
					if any(isinf(self.layers{lL}.es(:))), keyboard; end
                end
                end
			end
		end
		
		% CALCULATE THE GRADIENTS
		for lL = 2:self.nLayers-1
			[nX,nY,nM,nObs] = size(self.layers{lL}.fm);
			switch self.layers{lL}.type
			case 'conv'
				for jM = 1:self.layers{lL}.nFM
					es = self.layers{lL}.es(:,:,jM,:);
					for iM = 1:self.layers{lL-1}.nFM
						input = self.FLIPDIMS(self.layers{lL-1}.fm(:,:,iM,:));
						dEdFilter = convn(input,es,'valid')/nObs;
						self.layers{lL}.dFilter(:,:,iM,jM) = dEdFilter;
						if isnan(any(dEdFilter)), keyboard; end
					end
					self.layers{lL}.db(jM) = sum(es(:))/nObs;
				end
			case 'rect'
			case 'lcn'
			case 'pool'
            case 'hidden'
                if strcmp(self.layers{lL-1}.type,'hidden')
                    dE = self.layers{lL}.es;
                    fm=self.layers{lL-1}.fm;
                    dw=dE*fm'./nObs;
                    self.layers{lL}.dW = dw;
                    self.layers{lL}.db = mean(dE,2);
                else
                    dE = self.layers{lL}.es;
                    fm=self.layers{lL-1}.fm;
                    [nX,nObs,nM,~] = size(self.layers{lL}.fm);
                    dw=dE*reshape(fm,size(fm,1)*size(fm,2)*size(fm,3),size(fm,4))'./nObs;
                    dw=reshape(dw,size(dw,1),size(fm,1),size(fm,2),size(fm,3));
                    self.layers{lL}.dW = dw;
                    self.layers{lL}.db = mean(dE,2);
                end
            end
		end
		
		% GRADIENTS FOR OUTPUT LAYER WEIGHTS AND BIASES
		self.layers{end}.dW = outES*self.layers{end}.features'/nObs;
		self.layers{end}.db = mean(outES,2);
		
	end

	function self = updateParams(self)
		wPenalty = 0;
        self.layers{end}.W=self.layers{end}.W-self.layers{end}.lRate*self.layers{end}.dW;
        self.layers{end}.b=self.layers{end}.b-self.layers{end}.lRate*self.layers{end}.db;
		for lL = 2:self.nLayers-1
			switch self.layers{lL}.type
			case {'conv','output'}
				lRate = self.layers{lL}.lRate;
				for jM = 1:self.layers{lL}.nFM
					self.layers{lL}.b(jM) = self.layers{lL}.b(jM) - ...
					                    lRate*self.layers{lL}.db(jM);
					for iM = 1:self.layers{lL-1}.nFM
						self.layers{lL}.filter(:,:,iM,jM) = ...
						self.layers{lL}.filter(:,:,iM,jM) - ...
						lRate*(self.layers{lL}.dFilter(:,:,iM,jM)+wPenalty);
					end
                end
            case {'hidden'}
                 lRate = self.layers{lL}.lRate;
                 self.layers{lL}.b = self.layers{lL}.b - ...
					                    lRate*self.layers{lL}.db;
                 dW=reshape(self.layers{lL}.dW,size(self.layers{lL}.W));
                 self.layers{lL}.W =self.layers{lL}.W - lRate*dW;
			end
	    end
	end

	function out = calcAct(self,in,actFun)
		switch actFun
			case 'linear'
				out = self.stabilizeInput(in,1);

			case 'exp'
				in = self.stabilizeInput(in,1);
				out = exp(in);

			case 'sigmoid'
				in = self.stabilizeInput(in,1);
				out = 1./(1 + exp(-in));

			case 'softmax'
				in = self.stabilizeInput(in,1);
				maxIn = max(in, [], 2);
				tmp = exp(bsxfun(@minus,in,maxIn));
				out = bsxfun(@rdivide,tmp,sum(tmp,2));

			case 'tanh'
				in = self.stabilizeInput(in,1);
				out = tanh(in);
		end
	end

	function dAct = calcActDeriv(self,in,actFun)

		switch actFun
			case 'linear'
				dAct = ones(size(in));

			case 'exp';
				in = self.stabilizeInput(in,1);
				dAct = in;

			case 'sigmoid'
				in = self.stabilizeInput(in,1);
				dAct = in.*(1-in);
				
			case 'tanh'
				in = self.stabilizeInput(in,1);
				dAct = 1 - in.^2;
		end
	end

	function out = DOWN(self,data,stride)
		tmp = ones(stride(1),stride(2));
		tmp = tmp/prod(stride(:));
		out = convn(data,tmp,'valid');
		out = out(1:stride(1):end,1:stride(2):end,:,:,:);
	end
	
	function out = UP(self,data,scale);
		dataSz = size(data);
		idx = cell(numel(dataSz),1);
		for iD = 1:numel(dataSz)
			tmp = zeros(dataSz(iD)*scale(iD),1);
			tmp(1:scale(iD):dataSz(iD)*scale(iD)) = 1;
			idx{iD} = cumsum(tmp);
		end
		out = data(idx{:});
	end

	function out = ROT(self,out)
		out = out(end:-1:1,end:-1:1,:,:);
	end

	function out = FLIPDIMS(self,out)
		for iD = 1:numel(size(out))
			out = flipdim(out,iD);
		end
	end

	function self = makeBatches(self,data);
		nObs = size(data,4);
		nBatches = ceil(nObs/self.batchSize);
		idx = round(linspace(1,nObs+1,nBatches+1));

		for iB = 1:nBatches
			if iB == nBatches
				batchIdx{iB} = idx(iB):nObs;
			else
				batchIdx{iB} = idx(iB):idx(iB+1)-1;
			end
		end
		self.trainBatches = batchIdx;
    end
    
	% ASSES PREDICTIONS/ERRORS ON TEST DATA
	function [cost,pred] = test(self,data,targets,costFun)
		if notDefined('costFun') costFun=self.costFun; end

		for lL = 1:self.nLayers
			try
				self.layers{lL}.fm = [];
			catch
			end
		end
		self = self.fProp(data,targets);
		pred = self.netOut;
		cost = self.cost(targets,costFun);
	end

	function self = assessNet(self)
	%assessNet()
	%--------------------------------------------------------------------------
	%Utility function to assess the quality of current netork parameters and
	%store net, if necessary.
	%--------------------------------------------------------------------------
	
		if self.epoch > 1
			if self.trainCost(self.epoch) < self.bestTrainCost
				self.bestNet = self.layers;
				self.bestTrainCost = self.trainCost(self.epoch);
				self.stopEarlyCnt = 0;
			else
				self.stopEarlyCnt = self.stopEarlyCnt + 1;
			end
		else
			self.bestNet = self.layers; % STORE FIRST NET BY DEFAULT
		end
	end

	function printProgress(self,type)
	%printProgress(type)
	%--------------------------------------------------------------------------
	%Verbose utility function. <type> is the type of message to print.
	%--------------------------------------------------------------------------
		switch type
		case 'epoch'
			fprintf('Epoch: %i/%i',self.epoch,self.nEpoch);
		case 'trainCost'
			fprintf('\t%s: %2.3f\n',self.costFun,self.trainCost(self.epoch));
		case 'time'
			fprintf('\tTime: %g\n', toc);
		case 'xValCost'
			if ~self.stopEarlyCnt
				fprintf('\tCrossValidation Error:  %g (best net) \n',self.xValCost(self.epoch));
			else
				fprintf('\tCrossValidation Error:  %g\n',self.xValCost(self.epoch));
			end
		case 'gradCheck'
			netGrad = self.auxVars.netGrad;
			numGrad = self.auxVars.numGrad;
			gradFailed = self.auxVars.gradFailed;
			switch gradFailed
				case 1, gradStr = '(Failed)';
				otherwise, gradStr = '(Passed)';
			end
			fprintf('\tNetwork = %2.6f  \t Numerical = %2.6f  %s\n' ,netGrad,numGrad,gradStr);
		case 'save'
			fprintf('\nSaving...\n\n');
		end
	end

	function self = init(self,arch)
		arch = self.ensureArchitecture(arch);
		self.nLayers = numel(arch);
	    for lL = 1:numel(arch)
		    self.layers{lL}.type = arch{lL}.type;
		    switch arch{lL}.type
		    case 'input'
			    self.layers{lL}.dataSize = arch{lL}.dataSize;
			    self.layers{lL}.fmSize = arch{lL}.dataSize(1:2);
			    self.layers{lL}.nFM = arch{lL}.dataSize(3);

			case 'conv'
				if strcmp(arch{lL-1}.type,'input')
					nInY = arch{lL-1}.dataSize(1);
					nInX = arch{lL-1}.dataSize(2);
					nInFM = arch{lL-1}.dataSize(3);
				else
					nInX = self.layers{lL-1}.fmSize(2);
					nInY = self.layers{lL-1}.fmSize(1);
					nInFM = self.layers{lL-1}.nFM;
					
				end
				% INTERMEDIATE VARIABLES
				nFM = arch{lL}.nFM;
				filtSize = arch{lL}.filterSize;
				fmSize = [nInY,nInX] - filtSize + 1;
				fanIn = nInFM*prod(filtSize);
				fanOut = nFM*prod(filtSize);
				range = 2*sqrt(6/((fanIn + fanOut)));

				% INITIALIZE LAYER PARAMETERS
				self.layers{lL}.actFun = arch{lL}.actFun;
				self.layers{lL}.lRate = arch{lL}.lRate;
				self.layers{lL}.filterSize = filtSize;
				self.layers{lL}.fmSize = fmSize;
				self.layers{lL}.nFM = nFM;
				self.layers{lL}.fm = [];
				self.layers{lL}.filter = range*(rand([filtSize,nInFM,nFM])-.5);
				self.layers{lL}.dFilter = zeros(size(self.layers{lL}.filter));
				self.layers{lL}.b = zeros(nFM,1);
				self.layers{lL}.db = zeros(nFM,1);


			case 'subsample'
				% INTERMEDIATE VARIABLES
				stride = arch{lL}.stride;
				fmSize = floor(self.layers{lL-1}.fmSize./stride);
				nFM = self.layers{lL-1}.nFM;

				% INITIALIZE LAYER PARAMETERS
				self.layers{lL}.stride = stride;
				self.layers{lL}.fmSize = fmSize;
				self.layers{lL}.nFM = nFM;
				self.layers{lL}.fm = [];
				self.layers{lL}.b = zeros(nFM,1);
				self.layers{lL}.db = zeros(nFM,1);

            case 'hidden'
                if strcmp(arch{lL-1}.type,'hidden')
                    self.layers{lL}.type = 'hidden';
                    self.layers{lL}.lRate = arch{lL}.lRate;
                    range = 0.001;
                    self.layers{lL}.nFM=arch{lL}.nFM;
                    self.layers{lL}.W = (rand(arch{lL}.nFM,self.layers{lL-1}.nFM)-.5)*2*range;
                    self.layers{lL}.dW = zeros(size(self.layers{lL}.W));
                    self.layers{lL}.b = zeros(arch{lL}.nFM, 1);
                    self.layers{lL}.db = zeros(size(self.layers{lL}.b));
                    self.layers{lL}.actFun = arch{lL}.actFun;
                    self.layers{lL}.meanAct = zeros(1, arch{lL}.nFM);    
                else
                    self.layers{lL}.type = 'hidden';
                    self.layers{lL}.lRate = arch{lL}.lRate;
                    range = (1/((self.layers{lL-1}.nFM + self.layers{lL-1}.fmSize(2))));
                    self.layers{lL}.nFM=arch{lL}.nFM;
                    self.layers{lL}.W = (rand(arch{lL}.nFM,self.layers{lL-1}.fmSize(1),self.layers{lL-1}.fmSize(2),self.layers{lL-1}.nFM)-.5)*2*range;
                    self.layers{lL}.dW = zeros(size(self.layers{lL}.W));
                    self.layers{lL}.b = zeros(arch{lL}.nFM, 1);
                    self.layers{lL}.db = zeros(size(self.layers{lL}.b));
                    self.layers{lL}.actFun = arch{lL}.actFun;
                    self.layers{lL}.meanAct = zeros(1, arch{lL}.nFM);
%                     self.layers{lL}.a=arch{lL}.a;
                end    
			case 'output'

				nOut = arch{lL}.nOut;
				nFMIn = self.layers{lL-1}.nFM;
				nInFeats = prod([1,nFMIn]);
				range = 2*sqrt(6/(nInFeats + nOut));

				% ADJUST NETWORK OUTPUT LAYER PARAMETERS
				self.layers{lL}.nOut = nOut;
				self.layers{lL}.actFun = arch{lL}.actFun;
				self.layers{lL}.lRate = arch{lL}.lRate;
				self.layers{lL}.W = range*(rand(nOut,nInFeats)-.5);
				self.layers{lL}.dW = zeros(size(self.layers{lL}.W));
				self.layers{lL}.b = zeros(nOut,1);
				self.layers{lL}.db = zeros(nOut,1);
		    end
		end
	end


	function arch = ensureArchitecture(self,arch)
		if ~iscell(arch), error('<arch> needs to be a cell array of layer params');end
		if ~strcmp(arch{1}.type,'input'), error('define an input layer'); end
		if ~strcmp(arch{end}.type,'output'), error('define an output layer'); end

		% ENSURE LAYER-SPECIFIC PARAMS
		for lL = 1:numel(arch)
			lParams = fields(arch{lL});
			switch arch{lL}.type
			case 'input'
				if ~any(strcmp(lParams,'dataSize')), error('must provide data size');end

			case 'conv'
				if ~any(strcmp(lParams,'filterSize')) || isempty(arch{lL}.filterSize)
					arch{lL}.filterSize = [5 5];
				elseif numel(arch{lL}.filterSize) == 1;
					arch{lL}.filterSize = repmat(arch{lL}.filterSize,[1,2]);
				end
				if ~any(strcmp(lParams,'nFM'));
					arch{lL}.nFM = 10;
				end
				if ~any(strcmp(lParams,'lRate'));
					arch{lL}.lRate = .5;
				end
				if ~any(strcmp(lParams,'actFun'));
					arch{lL}.actFun = 'sigmoid';
				end

			case 'subsample'
				if ~any(strcmp(lParams,'stride')) || isempty(arch{lL}.stride);
					arch{lL}.stride = [1 1];
				elseif numel(arch{lL}.stride) == 1;
					arch{lL}.stride = repmat(arch{lL}.stride,[1,2]);
				end
			case 'output'
				if ~any(strcmp(lParams,'nOut'))
					error('must provide number of outputs');
				end
				if ~any(strcmp(lParams,'actFun'));
					arch{lL}.actFun = 'sigmoid';
				end
				if ~any(strcmp(lParams,'lRate'));
					arch{lL}.lRate = .5;
				end
			case 'rect'
			case 'lcn'
			case 'pool'
            case 'hidden'
                if ~any(strcmp(lParams,'nFM'));
					arch{lL}.nFM = 50;
				end
				if ~any(strcmp(lParams,'lRate'));
					arch{lL}.lRate = .5;
				end
			end
		end
	end

	function in = stabilizeInput(self,in,k);
		cutoff = log(realmin('single'));
		in(in*k>-cutoff) = -cutoff/k;
		in(in*k<cutoff) = cutoff/k;
	end


	function visLearning(self)
		try
			self.visFun(self);
		catch
			if ~isfield(self.auxVars,'printVisWarning')
				fprintf('\nWARNING: visualization failed.')
				self.auxVars.printVisWarning = true;
			end
		end
	end

	function save(self);
    	if self.verbose, self.printProgress('save'); end
		if ~isdir(self.saveDir)
			mkdir(self.saveDir);
		end
		fileName = fullfile(self.saveDir,sprintf('mlcnn.mat'));
		net = self.bestNet;
		save(fileName,'net');
	end
end % END METHODS
end % END CLASSDEF

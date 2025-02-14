class processor():
    '''Base on which all methods are built'''
    def __init__(self,data,method,latentModel,inputs, hank = False):
      self.data = data
      self.method = method
      self.lmodel = latentModel
      self.hank = hank
      import methods
      import sys
      if method in ["LR","LS"]:
        try:
            ncomp,metric = inputs
        except:
           print("Required inputs: [ncomp,metric]")
           sys.exit(1)
      elif method =="LC":
        try:
            ncomp,metric,cmodel,ccomp = inputs
        except ValueError:
           #print("Required inputs: [ncomp,metric,cmodel,ccomp]")
           raise ValueError
        ("Required inputs [ncomp,metric,cmodel,ccomp]")
      elif method =="LCON":
        try:
            ncomp,metric,cmodel = inputs
        except:
           print("Required inputs: [ncomp,metric,cmodel]")
           #sys.exit(1)
      elif method =="LEXP":
        try:
            cmodel,metric,cutoff,maxit= inputs
        except:
           print("Required inputs:[cmodel,metric,cutoff,maxit]")
           sys.exit(1)
    
    ##Prepare data
      if self.hank:
         from functions import hankelize
         X = hankelize(self.data,hank)
      else:
         X = self.data
      self.X = X
      ## Analyse Data
      if method in ["LR","LS"]:
        components = self.lmodel(X,ncomp)
        Methods = ["LR","LS"]
        from methods import ranker,scaler
        models = [ranker,scaler]
        model = models[Methods.index(method)]

        # fit model
        try:
           fitModel = model(components,X,metric)
        except:
           fitModel = model(components,X,metric)
            
        self.scores = fitModel.scores
        self.outs = fitModel.outComponents
      
      if method in ["LC"]:
         components = self.lmodel(X,ncomp)
         model = methods.clustering
         fitModel = model(components,X,ccomp,cmodel, metric)
         self.scores = fitModel.scores
         self.outs = fitModel.outComponents

      if method in ["LCON"]:
         components = self.lmodel(X,ncomp)
         model = methods.condensing
         fitModel = model(components,X,cmodel, metric)
         self.scores = fitModel.scores
         self.outs = fitModel.outComponents

      if method in ["LEXP"]:
         model = methods.LEXP
         fitModel = model(X,self.lmodel,cmodel,metric,cutoff,maxit)
         self.scores = fitModel.scores
         self.outs = fitModel.outComponents

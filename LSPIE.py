class processor():
    '''
    This is the main functional model of the LSPIE module.
    The list of model choices area: "LR" , "LS" , "LRS" , "LC" , "LCON" , "LEXP"
    '''
    def __init__(self,model,lvm,metric):
        import numpy as np
        self.lvm = lvm
        self.model = model
        self.metric = metric

        if model == "LR":
            self.fit = self.fitLR
        if model == "LS":
            self.fit = self.fitLS
        if model == "LRS":
            self.fit = self.fitLRS
        if model == "LCON":
            self.fit = self.fitLCON
        if model == "LC":
            self.fit = self.fitLC
        if model == "LEXP":
            self.fit = self.fitLEXP

    def fitLR(self,data,n,hank):
        from functions import hankelize
        import numpy as np
        if hank:
            self.X = hankelize(data,hank)
        else:
            self.X = data

        components = self.lvm(self.X,n)
        #self.components = components.T


        #find scores
        scores = []
        for component in components:
            score = self.metric(data,component)
            scores.append(score)
        
        #normalize and sort
        tot = sum([abs(met) for met in scores])
        Metrics = [abs(met) for met in scores]/tot
        sorts = Metrics.argsort()
        sorts = sorts[::-1]

        self.sorts = sorts
        self.scores = Metrics[sorts]*100
        self.components = np.array(components)[sorts]
        #return "this is ranking, not yet implemented, hankelization done"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fitLS(self,data,n,hank):
        from functions import hankelize
        import numpy as np
        if hank:
            self.X = hankelize(data,hank)
        else:
            self.X = data

        components = self.lvm(self.X,n)
        components = components


        #collect the scores
        metrics = []
        for component in components:
            metrics.append(self.metric(data,component))
        
        #normalize and sort
        tot = sum([abs(met) for met in metrics])
        Metrics = [abs(met) for met in metrics]/tot
        scores = Metrics
        #self.scores = scores

        scaledComps = []
        for i in range(len(scores)):
            scaled = components[i]*scores[i]/100
            scaledComps.append(scaled)
        
        
        #normalize and sort
        #tot = sum([abs(met) for met in scores])
        #Metrics = [abs(met) for met in scores]/tot
        #sorts = Metrics.argsort()
        #sorts = sorts[::-1]

        #self.sorts = sorts
        #self.scores = Metrics[sorts]*100
        #self.components = np.array(scaledComps)[sorts]
        self.scores = scores
        self.components = scaledComps
        #return "this is scaling, not yet implemented"

    def fitLRS(self,data,n,hank):
        from functions import hankelize
        import numpy as np
        if hank:
            self.X = hankelize(data,hank)
        else:
            self.X = data

        components = self.lvm(self.X,n)
        components = components


        #collect the scores
        metrics = []
        for component in components:
            metrics.append(self.metric(data,component))
        
        #normalize and sort
        tot = sum([abs(met) for met in metrics])
        Metrics = [abs(met) for met in metrics]/tot
        scores = Metrics
        #self.scores = scores

        scaledComps = []
        for i in range(len(scores)):
            scaled = components[i]*scores[i]/100
            scaledComps.append(scaled)
        
        
        #normalize and sort
        tot = sum([abs(met) for met in scores])
        Metrics = [abs(met) for met in scores]/tot
        sorts = Metrics.argsort()
        sorts = sorts[::-1]

        self.sorts = sorts
        self.scores = Metrics[sorts]*100
        self.components = np.array(scaledComps)[sorts]
        #return "this is scaling, not yet implemented"
  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
    def fitLC(self,data,n,cmodel,hank):
        from functions import hankelize
        import numpy as np

        f = 5
        if hank:
            self.X = hankelize(data,hank)
        else:
            self.X = data

        try: 
            components = self.lvm(self.X,f*n)
        except:
            components = self.lvm(self.X, min(np.shape(self.X)))

        #collect the scores
        metrics = []
        for component in components:
            metrics.append(self.metric(self.X,component))
        metrics = np.array(metrics)
        try:
            Yhat = cmodel.fit_predict(metrics)
        except:
            Yhat = cmodel.fit_predict(metrics.reshape(-1, 1))


        clusters = np.unique(Yhat)
        comps = []
        scores =[]
        Points_per_Cluster = []
        for cl in clusters:
            pts = np.where(Yhat == cl)[0]
            Points_per_Cluster.append([cl,len(pts)])
            component = components[pts]
            component = np.sum(component,axis = 0)
            comps.append(component.T)

            score = metrics[pts]
            score = np.sum(score,axis = 0)
            scores.append(score.T)

        tot = sum([abs(met) for met in scores])
        scores = [abs(met) for met in scores]/tot*100
        comps =np.array(comps)
        for i in range(len(comps)):
            comps[i]=comps[i]*scores[i]
        components = comps

        sorts = scores.argsort()
        sorts = sorts[::-1]

        self.components = components[sorts]
        self.scores = scores[sorts]
        #ok now: we have scores and a clustering model, how do we want this to work?
        #model has set number of components, .fit(), 
        #return "this is clustering, not yet implemented"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def fitLCON(self,data,cmodel,mincomps,hank):
        from functions import hankelize
        import numpy as np
        if hank:
            X = hankelize(data,hank)
        else:
            X = data

        try: 
            n = min(np.shape(X)) 
            components = self.lvm(X,10*n)
        except:
            components = self.lvm(X, min(np.shape(X)))

        #collect the scores
        metrics = []
        for component in components:
            metrics.append(self.metric(data,component))
        metrics = np.array(metrics)
        try:
            Yhat = cmodel.fit_predict(metrics)
        except:
            Yhat = cmodel.fit_predict(metrics.reshape(-1, 1))


        clusters = np.unique(Yhat)
        comps = []
        scores =[]
        Points_per_Cluster = []
        for cl in clusters:
            pts = np.where(Yhat == cl)[0]
            Points_per_Cluster.append([cl,len(pts)])
            component = components[pts]
            component = np.sum(component,axis = 0)
            comps.append(component.T)

            score = metrics[pts]
            score = np.sum(score,axis = 0)
            scores.append(score.T)

        tot = sum([abs(met) for met in scores])
        scores = [abs(met) for met in scores]/tot*100
        comps =np.array(comps)
        for i in range(len(comps)):
            comps[i]=comps[i]*scores[i]
        components = comps

        sorts = scores.argsort()
        sorts = sorts[::-1]

        if len(components)>=mincomps:
            self.components = components[sorts]
            self.scores = scores[sorts]
        else:
            print("Too Few Components Found")
            from sklearn.cluster import Birch ,DBSCAN # type: ignore
            cmodel = Birch(n_clusters = mincomps)
            try: 
                components = self.lvm(self.X,f*n)
            except:
                components = self.lvm(self.X, min(np.shape(X)))

            #collect the scores
            metrics = []
            for component in components:
                metrics.append(metric(self.X,component))
            metrics = np.array(metrics)
            try:
                Yhat = cmodel.fit_predict(metrics)
            except:
                Yhat = cmodel.fit_predict(metrics.reshape(-1, 1))


            clusters = np.unique(Yhat)
            comps = []
            scores =[]
            Points_per_Cluster = []
            for cl in clusters:
                pts = np.where(Yhat == cl)[0]
                Points_per_Cluster.append([cl,len(pts)])
                component = components[pts]
                component = np.sum(component,axis = 0)
                comps.append(component.T)

                score = metrics[pts]
                score = np.sum(score,axis = 0)
                scores.append(score.T)

            tot = sum([abs(met) for met in scores])
            scores = [abs(met) for met in scores]/tot*100
            comps =np.array(comps)
            for i in range(len(comps)):
                comps[i]=comps[i]*scores[i]
            components = comps

            sorts = scores.argsort()
            sorts = sorts[::-1]

        self.components = components[sorts]
        self.scores = scores[sorts]              
        

        return 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fitLEXP(self,data,maxcomp,cutoff,hank):
        from functions import hankelize
        import numpy as np
        minscore = 1000
        j = 2

        if hank:
            X = hankelize(data,hank)
        else:
            X = data

        while minscore > cutoff and j < maxcomp:
            components = self.lvm(X,j)
                    #find scores
            scores = []
            for component in components:
                score = self.metric(data,component)
                scores.append(score)
            #print(scores)
            tot = sum([abs(met) for met in scores])
            scores = [abs(met) for met in scores]/tot
            components =np.array(components)
            for i in range(len(components)):
                components[i]=components[i]*scores[i]
            components = components
            minscore = min(scores)
            #print(minscore)
            j = j+1

        sorts = scores.argsort()
        sorts = sorts[::-1]

        self.components = components[sorts]
        self.scores = scores[sorts]
        return 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   


        

                    


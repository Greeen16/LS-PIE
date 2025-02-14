# METHOD 1
class ranker():
    '''Takes inputted components, the original signal, and a metric and returns the scores and ranked components '''
    def innerProduct(x,y):
        from numpy import dot, sum
        return sum(dot(y,x))
    def __init__(self,components,signal, metric = innerProduct):
        import numpy as np
        self.components = components
        self.signal = signal
        self.metric = metric

        #collect the scores
        metrics = []
        for component in components:
            metrics.append(metric(signal,component))
        
        #normalize and sort
        tot = sum([abs(met) for met in metrics])
        Metrics = [abs(met) for met in metrics]/tot
        sorts = Metrics.argsort()
        sorts = sorts[::-1]
        
        self.sorts = sorts
        self.scores = Metrics[sorts]*100
        self.outComponents = np.array(components)[sorts]
##################################################################
##################################################################

#Method 2
class scaler():
    ''' Takes inputted components, the original signal, and a metric and returns the scores and scaled ranked components'''
    def innerProduct(x,y):
        from numpy import dot, sum
        return sum(dot(y,x))
        
    def __init__(self,components,signal, metric = innerProduct):
            import numpy as np
            self.components = components
            self.signal = signal
            self.metric = metric

            #collect the scores
            metrics = []
            for component in components:
                metrics.append(metric(signal,component))
            
            #normalize and sort
            tot = sum([abs(met) for met in metrics])
            Metrics = [abs(met) for met in metrics]/tot
            scores = Metrics
            self.scores = scores

            scaledComps = []
            for i in range(len(self.scores)):
                scaled = components[i]*self.scores[i]/100
                scaledComps.append(scaled)
            self.outComponents = scaledComps 
##################################################################
##################################################################

#Method 3
class rankedScaler():
    ''' Takes inputted components, the original signal, and a metric and returns the scores and scaled ranked components'''
    def innerProduct(x,y):
        from numpy import dot, sum
        return sum(dot(y,x))
        
    def __init__(self,components,signal, metric = innerProduct):
            import numpy as np
            self.components = components
            self.signal = signal
            self.metric = metric

            ranking = ranker(self.components,self.signal,self.metric)
            self.scores = ranking.scores
            rankedComps = ranking.outComponents

            scaledComps = []
            for i in range(len(self.scores)):
                scaled = rankedComps[i]*self.scores[i]/100
                scaledComps.append(scaled)
            self.outComponents = scaledComps   
##################################################################
##################################################################

#Method 4    
class clustering():
    '''Takes inputted components, the original signal, and a metric and returns the clustered components and a score '''

    def innerProduct(x,y):
        from numpy import dot, sum
        return sum(dot(y,x))
    def __init__(self,components,signal,n_comp,clusteringModel = None, metric = None):
        from sklearn.cluster import Birch ,DBSCAN # type: ignore
        import numpy as np
        
        if clusteringModel == None:
            clusteringModel = Birch(threshold=0.1, n_clusters=n_comp)
            #print("Model is Birch")

        else:
            clusteringModel = clusteringModel(n_clusters = n_comp)
        self.clusteringModel = clusteringModel
        if metric == None:
            Yhat = clusteringModel.fit_predict(components)
        else:
            scores = metric(signal,components)
            print(np.shape(scores))
            Yhat = clusteringModel.fit_predict(scores.reshape(-1, 1))
        
        clusters = np.unique(Yhat)
        Comps = []
        Points_per_Cluster = []
        for cl in clusters:
            pts = np.where(Yhat == cl)[0]
            Points_per_Cluster.append([cl,len(pts)])
            component = components[pts]
            component = np.sum(component,axis = 0)
            Comps.append(component.T)

        self.Points_Per_Cluster = Points_per_Cluster
        PTS = []
        for i in self.Points_Per_Cluster:
            PTS.append(i[1])

        PTS = np.array(PTS)
        n = np.sum(PTS)
        self.scores = PTS/n*100
        
        scaledComps = []
        for i in range(len(self.scores)):
            scaled = (Comps[i]/np.max(Comps[i]))*self.scores[i]/100
            scaledComps.append(scaled)
        self.outComponents = scaledComps    
##################################################################
##################################################################

#Method 5    
class condensing():
    '''
    Applies the Condensing Functionality of LS-PIE
    '''

    def innerProduct(x,y):
        from numpy import dot, sum
        return sum(dot(y,x))
    def __init__(self,components,signal,clusteringModel = None, metric = None):
        from sklearn.cluster import Birch ,DBSCAN  # type: ignore
        import numpy as np
        if clusteringModel == None:
            clusteringModel = DBSCAN(eps=0.1, min_samples=10)
            #print("Model is DBSCAN")
        else:
            clusteringModel = clusteringModel
        self.clusteringModel = clusteringModel 
        if metric == None:
            Yhat = clusteringModel.fit_predict(components)
        else:
            scores = metric(signal,components)
            Yhat = clusteringModel.fit_predict(scores)
        
        clusters = np.unique(Yhat)
        Comps = []
        Points_per_Cluster = []
        for cl in clusters:
            pts = np.where(Yhat == cl)[0]
            Points_per_Cluster.append([cl,len(pts)])
            component = components[pts]
            component = np.sum(component,axis = 0)
            Comps.append(component.T)

        self.Points_Per_Cluster = Points_per_Cluster
        PTS = []
        for i in self.Points_Per_Cluster:
            PTS.append(i[1])

        PTS = np.array(PTS)
        n = np.sum(PTS)
        self.scores = PTS/n*100
        
        scaledComps = []
        for i in range(len(self.scores)):
            scaled = (Comps[i]/np.max(Comps[i]))*self.scores[i]/100
            scaledComps.append(scaled)
        self.outComponents = scaledComps 

# Method 6
     
class LEXP():
    '''
    Applies LEXP
    '''
    def __init__(self, X, LVM = None,clusteringModel = None, metric = None, cutoff = None, maxit = 100):
        from functions import prebuiltICA, nonCluster,expDot,norm2
        import numpy as np

        if LVM == None:
            self.LVM = prebuiltICA
        else:
            self.LVM = LVM

        if clusteringModel == None:
            self.clusteringModel = nonCluster
        else:
            self.clusteringModel = clusteringModel

        if metric == None:
            self.metric = expDot
        else:
            self.metric = metric

        if cutoff == None:
            self.cutoff = 0.1
        else:
            self.cutoff = cutoff
        
        minScore = 10
        
        n_components = 2

        while minScore>= cutoff and n_components<maxit:
            components = LVM(X,n_components)

            components = clusteringModel(components)

            scores = metric(X,components)
            print(scores)
            norm = norm2(scores)
            print(norm)
            minScore = min(norm)
            print(minScore)
            n_components +=1

        scores = np.sort(norm)
        sorts =scores.argsort()
        sorts = sorts[::-1]

        self.scores = scores[sorts]*100
        self.components = np.array(components)[sorts]

        self.outComponents = self.components

 
        

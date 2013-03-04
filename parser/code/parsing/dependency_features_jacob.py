import sys
import numpy as np
from scipy.sparse import lil_matrix
from dependency_reader import *
from itertools import chain, combinations

class DependencyFeatures():
    '''
    Dependency features class
    '''
    def __init__(self, use_lexical = False, use_distance = False, use_contextual = False):
        self.feat_dict = {}
        self.n_feats = 0

    def create_dictionary(self, instances):
        '''Creates dictionary of features (note: only uses supported features)'''
        self.feat_dict = {}
        self.n_feats = 0
        for instance in instances:
            nw = np.size(instance.words)-1
            heads = instance.heads
            for m in range(1, nw+1):
                h = heads[m]
                self.create_arc_features(instance, h, m, True)

        print "Number of features: {0}".format(self.n_feats)


    def create_features(self, instance):
        '''Creates arc features from an instance.'''
        nw = np.size(instance.words)-1
        feats = np.empty((nw+1, nw+1), dtype=object)
        for h in range(0,nw+1): 
            for m in range(1,nw+1):
                if h == m:
                    feats[h][m] = []
                    continue
                feats[h][m] = self.create_arc_features(instance, h, m)

        return feats

    def create_arc_features(self,instance,h,m,add=False):
        '''
        Create features for arc h-->m
        This is the function you should modify to do the project
        '''
        ff = []
        k = 0 #feature counter

        distance_features = True
        lexical_features = True
        bilexical_features = True
        context_features = True

        ## Head pos, modifier pos
        f = self.getF((k,instance.pos[h], instance.pos[m]),add)
        ff.append(f)
        k+=1

        ## your features go here
        if distance_features:
            dist = m - h
            if dist > 10: dist = 10
            elif dist > 5: dist = 5
            if dist < -10: dist = -10
            elif dist < -5: dist = -5
            
            f = self.getF((k,dist),add)
            ff.append(f)

        if lexical_features:
            ff.append(self.getF((k,instance.words[h],instance.pos[m]),add))
            k+=1
            ff.append(self.getF((k,instance.pos[h],instance.words[m]),add))
            k+=1

        if bilexical_features:
            ff.append(self.getF((k,instance.words[h],instance.words[m]),add))
            k+=1

        if context_features:
            nw = np.size(instance.words)
            #previous tag from head
            if h == 0: hpp = "__START__"
            else: hpp = instance.pos[h-1]
            #next tag from head
            if h == nw-1: hpn = "__END__"
            else: hpn = instance.pos[h+1]
            #previous tag from modifier
            if m == 0: mpp = "__START__"
            else: mpp = instance.pos[m-1]
            #next tag from modifier
            if m == nw-1: mpn = "__END__"
            else: mpn = instance.pos[m+1]

            context_features = [hpp,hpn,mpp,mpn]
            feat_it = chain.from_iterable(combinations(context_features,r) for r in range(len(context_features)+1))
            feat_it.next() #burn the first one
            for feat in feat_it:
                ff.append(self.getF((k,instance.pos[h],instance.pos[m])+feat,add))
                k = k+1

        return(ff)

    def getF(self, feats, add=True):
        return self.lookup_fid(feats,add)

    def lookup_fid(self, fname, add=False):
        '''Looks up dictionary for feature ID.'''
        if not fname in self.feat_dict:
            if add:
                fid = self.n_feats
                self.n_feats += 1
                self.feat_dict[fname] = fid
                return fid
            else:
                return -1
        else:
            return self.feat_dict[fname]

    def compute_scores(self, feats, weights):
        '''Compute scores by taking the dot product between the feature and weight vector.''' 
        nw = np.size(feats, 0) - 1
        scores = np.zeros((nw+1, nw+1))
        for h in range(nw+1):
            for m in range(nw+1):
                if feats[h][m] == None:
                    continue
                scores[h][m] = sum([weights[f] for f in feats[h][m] if f>=0])
        return scores



import numpy as np
from subprocess import check_output
import tempfile,os,sys
from collections import Counter
import struct,math
from keras.utils import np_utils

class ASR(object):
    """
    ASR functions using HTK, julius
    path: main path 
    dic: dictionay path
    lm: language_model path
    hlist: the list of monophones
    julius: the path of julius
    hresults: the path of hresults
    priors_path: the path of prior probabilities of HMM states
    testmfclist: the path of the mfc file list for test set
    testdnnlist: the path of the dnn stateProbs for test set
    setID: the set ID for cross validation
    prob_path: the folder where stateProbs will be saved
    results_mlf: the path of recognized mlf
    label_mlf: real mlf
    steps: how many steps the networks process each time
    ahead: how many steps is the future information earlier than present information
    """
    def __init__(self, 
    path, dic, lm, hlist, julius, hresults, priors_path,
    testmfclist, testdnnlist, setID, prob_path, 
    results_mlf, label_mlf, opt, model,
    steps=20,N_cell=512):
        self.path=path
        self.dic=dic
        self.lm=lm
        self.hlist=hlist
        self.julius=julius
        self.hresults=hresults
        self.priors_path=priors_path
        self.testmfclist=testmfclist
        self.testdnnlist=testdnnlist
        self.setID=setID
        self.prob_path=prob_path
        self.results_mlf=results_mlf
        self.label_mlf=label_mlf
        self.opt=opt
        self.model=model
        self.steps=steps
        self.N_cell=N_cell
    ##htk file reader/writer
    #htk writer
    def write_htk(self, features,outputFileName,fs=100,dt=9):
        """
        in: file name
        out: out[0] is the data
        """
        sampPeriod = 1./fs
        pk =dt & 0x3f
        features=np.atleast_2d(features)
        if pk==0:
            features =features.reshape(-1,1)
        with open(outputFileName,'wb') as fh:
            fh.write(struct.pack(">IIHH",len(features),int(sampPeriod*1e7),features.shape[1]*4,dt))
            features=features.astype(">f")
            features.tofile(fh)
    #htk reader
    def read_htk(self, inputFileNmae, framePerSecond=100):
        """
        in: file name
        """
        kinds=['WAVEFORM','LPC','LPREFC','LPCEPSTRA','LPDELCEP','IREFC','MFCC','FBANK','MELSPEC','USER','DISCRETE','PLP','ANON','???']
        with open(inputFileNmae, 'rb') as fid:
            nf=struct.unpack(">l",fid.read(4))[0]
            fp=struct.unpack(">l",fid.read(4))[0]*-1e-7
            by=struct.unpack(">h",fid.read(2))[0]
            tc=struct.unpack(">h",fid.read(2))[0]
            tc=tc+65536*(tc<0)
            cc='ENDACZK0VT'
            nhb=len(cc)
            ndt=6
            hb=list(int(math.floor(tc * 2 ** x)) for x in range(- (ndt+nhb),-ndt+1))
            dt=tc-hb[-1] *2 **ndt
            if any([dt == x for x in [0,5,10]]):
                aise('NOt!')
            data=np.asarray(struct.unpack(">"+"f"*int(by/4)*nf,fid.read(by*nf)))
            d=data.reshape(nf,int(by/4))
        t=kinds[min(dt,len(kinds)-1)]
        return (d,fp,dt,tc,t)
    #htk getstateprobs using DNN model 
    def GetStateProbs(self, get_out):
        #load state priors
        priors=dict([line.split() for line in open(self.priors_path).readlines()])
        priors=dict([(int(a),float(b)) for a, b in priors.items()])
        #turn into 1D vector in order
        priors=np.array([priors[i] for i in range(len(priors))])
        #get file list
        fnames=open(self.testmfclist).readlines()
        for name in fnames:
            data=self.read_htk(name[:-1])[0]
            #make data
            
            lists=self.per_sentence(data,seq_Y='0',steps=2000)
            probs=[]
            for l in lists:
                out=get_out(l[0])
                probs.append(out)
            #turn into likelihoods
            log_liks=np.log10(np.vstack(probs)/priors)
            path=self.prob_path+name.split("/")[-1][:-4]+"llk"
            self.write_htk(log_liks,path)
            
    def RecogWithStateProbs(self, get_out, s=2,pl=2):
        self.GetStateProbs(get_out)
        cmd='echo | %s -filelist %s -hlist %s -h %s -nlr %s -v %s %s -lmp %s %s %s' %(self.julius, self.testdnnlist, self.hlist, self.model, self.lm, self.dic, self.opt, s, pl,"-input outprob")
        result=check_output(cmd,shell=1).split("\n")
        phone=["#!MLF!#\n"]
        f=open(self.testdnnlist,"r")
        train=f.readlines()
        f.close()
        i=0
        #take result lines
        setname=self.testdnnlist.split("/")[-1].split("mfc")[0]
        for r in result:
            if 'sentence1' in r:
                fn='"*/'+train[i].split("/")[-1][:-5]+'.rec"\n'
                rec=(("s_s"+r.split("<s>")[1]).replace("</s>","s_e")).replace(" ","\n")+"\n.\n"
                phone.append(fn+rec)
                i+=1
        #write mlf
        fw=open(self.results_mlf,"w")
        for p in phone:
            fw.write(p)
        fw.close()
        #run HTK HResults
        cmd=self.hresults+' -A -z ::: -I %s -e ::: s_s -e ::: s_e %s %s' %(self.label_mlf,self.hlist,self.results_mlf)
        acc=check_output(cmd,shell=1)
        PER=100-float(acc.split("\n")[-3].split(" ")[2].split("=")[1])
        return PER

    def per_sentence(self, seq_X, seq_Y='0',steps=20):
        """M2MGRU data generator 
        Args:
          seq_X: a sequence of vectors for one sentence (len, input_Dim)
          seq_Y: a sequence of vectors for one sentence (len, output_Dim)
          time_step: the length of history
        Returns: a dictionary dic which consists of two/three terms 
          in_present: (1, steps, input_dim)
          in_future:  (1, steps+ahead, input_dim)
          labels: (1, steps, output_dim)
        note:
      
        """
        D_input=seq_X.shape[1]
        D_output=120
        sen_len=len(seq_X)
        lists=[]
        #index
        idx_present=0
        #first part
        #remaining parts
        while idx_present < sen_len:
            if type(seq_Y)!=type('0'):
                lists.append([seq_X[idx_present:idx_present+steps].reshape((1,-1,D_input)).astype("float32")[:,:,:39],np_utils.to_categorical(seq_Y[idx_present:idx_present+steps],D_output).astype("int16"),seq_X[idx_present:idx_present+steps][:,39:].astype("float32")])
            else:
                lists.append([seq_X[idx_present:idx_present+steps].reshape((1,-1,D_input)).astype("float32")[:,:,:39]])
            idx_present+=steps
        return lists
    
    def total_sentences(self,fname,vali_size=10):
        """lstm data generator 
        return:
          total sentences tensors
        """
        self.train_data=[]
        for x,y in zip(np.load(fname['feature']),np.load(fname['label'])):
            self.train_data.append(self.per_sentence(x,y,self.steps))
        self.vali_data=self.train_data[:int(len(self.train_data)/vali_size)]
        self.train_data=self.train_data[int(len(self.train_data)/vali_size):]
    #shuffle several lists together
    def shufflelists(self):
        ri=np.random.permutation(len(self.train_data))
        self.train_data=[self.train_data[i] for i in ri]

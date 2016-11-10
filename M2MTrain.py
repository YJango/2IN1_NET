import sys
from M2MASR import ASR
from M2MGRU import M2MGRU
import numpy as np
import theano.tensor as T
import lasagne
import theano

#processing training process
def merges(inver,one):
    return np.concatenate((np.load(inver),np.load(one)))

def processing(b,n_samples,times,acc,loss,accvali,lossvali):
    sys.stdout.write('Epoch:%2.2s(%4.4s) | Best acc:%6.6s loss:%6.6s | Cur acc:%6.6s loss:%6.6s\r' %(times,round((float(b)/n_samples)*100,1),round(float(accvali),4),round(float(lossvali),4),round(float(acc),4),round(float(loss),4)))
    sys.stdout.flush()
#write log file
def writer(fname,acc,loss,accvali,lossvali,times):
    f=open(fname+'.txt','a')
    f.write('Epoch:%2.2s | Best acc:%6.6s loss:%6.6s | Cur acc:%6.6s loss:%6.6s\n' %(times,round(float(accvali),4),round(float(lossvali),4),round(float(acc),4),round(float(loss),4)))
    f.close()
#vali loss and acc
def vali_loss_acc(vali,get_out,N_cell):
    predicts=[]
    targets=[]
    P=T.matrix('P')
    R=T.matrix('R')
    for i in range(len(vali)):
        #hidden states
        for l in vali[i]:
            #(l[0].shape,l[1].shape)
            out=get_out(l[0])
            predicts.append(out)
            targets.append(l[1])
    
    loss= T.mean(lasagne.objectives.categorical_crossentropy(P, R))       
    acc = T.mean(lasagne.objectives.categorical_accuracy(P, R))
    performance=theano.function([P,R],[loss,acc])
    return performance(np.vstack(predicts),np.vstack(targets))

def trainwithPER(setID,typ='teacher',learning_rate=1e-4, drop_out=0.4,Layers=[2,1,2], N_hidden=2048, N_cell=2048, steps=20, L2_lambda=1e-4,patience=3, continue_train=0, evalPER=0,lamda=0.5):
    N_EPOCHS = 100
    np.random.seed(55)
    typ2=typ
    path="/home/jango/distillation"
    dic=path+"/shared_babel/phone.dic"
    lm=path+"/shared_babel/pdnet"
    hlist=path+"/shared_babel/monophones1"
    opt="-sb 160 -b2 60 -s 1000 -m 4000 -quiet -1pass"
    julius='julius-4.3.1'
    hresults='HResults'
    priors_path=path+"/%s/StatPrior%s_train" %(typ2,setID)
    testmfclist=path+"/%s/feature/list%s/testmfc.list" %(typ,setID)
    testdnnlist=path+"/%s/feature/list%s/testdnn.list" %(typ,setID)
    prob_path=path+"/%s/StatePro%s/" %(typ,setID)
    results_mlf=path+"/%s/LSTMRec/rec.mlf" %typ
    
    label_mlf=path+'/shared_babel%s/mlf%s/alignedtest.mlf' %(typ[:3],setID)
    model=path+"/%s/HMM%s/hmmdefs" %(typ2,setID)
    train_data={'feature':path+"/%s/LSTMFile%s/%s_train_lstm.npy" %(typ,setID,typ[:3]),
                'label':path+"/%s/LSTMFile%s/%s_train_target_lstm.npy" %(typ2,setID,typ2[:3])}
    #where to store weights
    fname='%s/%s/LSTMWeight%s/Pretrain2in1_%s_N%s_D%s_L%s_C%s_%s_S%s_l%s' %(path,typ,setID,typ[:3],N_hidden,drop_out,L2_lambda,N_cell,Layers,steps,lamda)
    print(fname)
    inver='%s/%s/LSTMWeight%s/SBGRU_tea_N2048_D0.3_L0.001_C1024_[2, 1, 2].npy' %(path,'student',setID)
    one='%s/%s/LSTMWeight%s/FULL_tea_N2048_D0.3_L0.001_C1024_[2, 2, 2]_S2000.npy' %(path,'teacher',setID)
    #instances
    asr=ASR(path, dic, lm, hlist, julius, hresults, priors_path,testmfclist, testdnnlist, setID, prob_path, results_mlf, label_mlf, opt, model,steps=steps,N_cell=N_cell)
    #data maker
    asr.total_sentences(train_data)
    
    ff=M2MGRU(learning_rate=learning_rate, drop_out=drop_out, Layers=Layers, N_hidden=N_hidden, N_cell=N_cell, D_input=asr.train_data[0][0][0].shape[2],D_art=48,  D_out=120, Task_type='classification', L2_lambda=L2_lambda, _EPSILON=1e-15,lamda=lamda)
    
    ff.loader(merges(inver,one))
    #whether to retrain
    if continue_train:
        ff.loader(np.load(fname+'.npy'))
    if evalPER:
        ff.loader(np.load(fname+'.npy'))
        PER=asr.RecogWithStateProbs(ff.get_out,2,2)
        print(PER,lamda)
        return 0
    #init for training 
    epoch = 0
    p = 1
    print(asr.RecogWithStateProbs(ff.get_out,2,2))
    lossvali=100
    accvali=0
    acc=0
    loss=100
    n_sentences=len(asr.train_data)
    print('n_train:',n_sentences)
    asr.shufflelists()
    # start to train
    for epoch in range(N_EPOCHS):
        for i in range(n_sentences):
            #hidden states
            for l in asr.train_data[i]:
                ff.train(l[0],l[1],l[2])
            processing(i,n_sentences,epoch,acc,loss,accvali,lossvali)
        asr.shufflelists()
        #current preformences
        loss, acc = vali_loss_acc(asr.vali_data,ff.get_out,N_cell)
        #whether to change learning rate
        if loss<=lossvali:
            #save weights
            ff.saver(fname)
            #update the best acc and loss
            lossvali=loss
            accvali=acc
            writer(fname,acc,loss,accvali,lossvali,epoch)
            #reset patience
            p=1
        else:
            if p>patience:
                writer(fname,acc,loss,accvali,lossvali,epoch)
                f=open(fname+'.txt','a')
                ff.loader(np.load(fname+'.npy'))
                f.write('\nEpoch: %s | Best acc:%s loss: %s | PER:%s\n' %(epoch, round(float(accvali),4), round(float(lossvali),4), asr.RecogWithStateProbs(ff.get_out,2,2)))
                f.write('fine-tuning with lr %s\n' %(5e-6))
                f.close()
                return 0
            p+=1

for ll in [6]:
    trainwithPER(str(ll),typ='teacher',learning_rate=9e-5,drop_out=0.3,Layers=[2,1,2,2,2,2], N_hidden=2048, N_cell=1024, steps=2000, L2_lambda=1e-3,patience=3, continue_train=1,evalPER=0,lamda=0.7)
    trainwithPER(str(ll),typ='teacher',learning_rate=5e-6,drop_out=0.3,Layers=[2,1,2,2,2,2], N_hidden=2048, N_cell=1024, steps=2000, L2_lambda=1e-3,patience=2, continue_train=1,evalPER=0,lamda=0.7)

#for ll in [0.3]:
    #trainwithPER(str(0),typ='teacher',learning_rate=9e-5,drop_out=0.3,Layers=[2,1,2,2,2,2], N_hidden=2048, N_cell=1024, steps=2000, L2_lambda=1e-3,patience=3, continue_train=0,evalPER=0,lamda=ll)
    #trainwithPER(str(1),typ='teacher',learning_rate=5e-6,drop_out=0.3,Layers=[2,1,2,2,2,2], N_hidden=2048, N_cell=1024, steps=2000, L2_lambda=1e-3,patience=2, continue_train=0,evalPER=1,lamda=ll)

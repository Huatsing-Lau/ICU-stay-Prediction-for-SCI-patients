# -*- coding: utf-8 -*-
"""

Created on Sun May 12 17:38:51 2019
# FCN网络用语预测N年生存概率（二分类或多分类问题。）
@author: liuhuaqing
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import metrics
#from sklearn.preprocessing import label_binarize
import time
import os

import tensorflow as tf #version: tensorflow 2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import L1L2


def lossfun_C(Ygt_khot,Ypre_logits,name=None):
    #E1 = -y_label_khot * np.log(y_logit_softmax) - (1-y_label_khot) * np.log(1-y_logit_softmax)
    loss = tf.reduce_mean( -(1-Ygt_khot)*tf.log(1.01-tf.nn.softmax(Ypre_logits)),
                          name='unC_loss_cross_entropy_softmax' )
    return loss


def generate_arrays_from_arrays(YX_array):
    while True:
        Y = YX_array[:,0]
        X = YX_array[:,1:]
        X += np.random.normal(loc=0.0, scale=1e-2, size=X.shape)
        yield ( np.array(X), np.array(Y) )

    

class dnn(object):
    def __init__(self, Data=None,
                 checkpoint_path = "training_record/cp.ckpt",
                 input_node=None, output_node=None, hidden_layers_node=None, 
                 max_epoch=10000,
                 activation='relu',
                 optimizer=tf.keras.optimizers.Adam(), 
                 loss_function=tf.keras.losses.MeanAbsoluteError(),
                 L1_reg=0.2, L2_reg=0.2, 
                 dropout_rate=0.2, seed=42):
        """dnn(Deep Survival Neural Network) Class Constructor

        Parameters
        ----------
        X : np.array
            Input data with covariate variables.
        label : np.array
            Input data with categorical variables.
        input_node : int
            Number of input layer.
        hidden_layers_node : list
            Number of nodes in hidden layers of neural network.
        output_node : int
            Number of output layer.
        learning_rate : float
            Learning rate.
        learning_rate_decay : float
            Decay of learning rate.
        activation : string
            Type of activation function. The options include `relu`, `sigmoid` and `tanh`.
        L1_reg : float
            Parameter of L1 regularizate item.
        L2_reg : float
            Parameter of L2 regularizate item.
        optimizer : string
            Type of optimize algorithm. The option include `sgd` and `adam`.
        dropout_rate : float
            Probability of dropout.
        seed : int
            Random state settting.

        Returns
        -------
        `dnn class`
            An instance of `dnn class`.

        Examples
        --------
        >>> train_data = load_data()
        >>> train_X = train_data['x']
        >>> train_y = {'e': train_data['e'], 't': train_data['t']}
        >>> model = dnn(X, y, 117, [64, 32, 8], 1)
        """
        # Prepare data
        self.Data = Data  
        self.checkpoint_path = checkpoint_path
        self.max_epoch = max_epoch
        self.input_node = input_node
        self.output_node = output_node
        self.hidden_layers_node = hidden_layers_node
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.L1_reg = L1_reg
        self.L2_reg = L1_reg
        self.best_threshold = 0.5
        
        if input_node is None:
            input_node = Data['X'].shape[1]
        if output_node is None:
            output_node = np.unique(Data['Y']).size
            
        self.model = DNN(input_node,output_node,hidden_layers_node,
                         activation,dropout_rate,
                         optimizer=tf.keras.optimizers.Adam(),
                         loss_function=self.loss_function,
                         L1_reg=self.L1_reg, L2_reg=self.L2_reg)
        
    def train(self, X_train=None, Y_train=None, 
              X_val=None, Y_val=None, 
              num_epoch=1000, 
              plot_loss_curve=True, 
              plot_auc_curve=True):
        """Training dnn.

        Parameters
        ----------
        num_epoch : int
            Number of epoch.
        iteration : int
            Number of iteration, after which printing information of training processes.
            Default -1, means keep silence.
        plot_train_loss : bool
            Does plot curve of loss value during training.
        plot_train_auc : bool
            Dose plot curve of CI on train set during training.

        Returns
        -------
        None

        Examples
        --------
        >>> model.train(X_train, Y_train, num_epoch=2500, plot_loss_curve=True, plot_auc_curve=True)
        """
        
        if X_train is None:
            X_train = self.Data['X_train']
        if Y_train is None:
            Y_train = self.Data['Y_train']
        if X_val is None:
            X_val = self.Data['X_test']
        if Y_val is None:
            Y_val = self.Data['Y_test']

        # Train
# =============================================================================
# #         if os.path.isfile(self.checkpoint_path):
#         if os.path.isdir(os.path.dirname(self.checkpoint_path)):
# #             import pdb
# #             pdb.set_trace()
#             import shutil
#             shutil.rmtree(os.path.dirname(self.checkpoint_path))#删除文件夹
# =============================================================================

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         save_freq='epoch')

        reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=1e-6, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
        

        YX_train = np.concatenate((Y_train[:,np.newaxis],X_train), axis=1)
        import pdb
        history = self.model.fit_generator(
                generator = generate_arrays_from_arrays(YX_train),
                steps_per_epoch = 2,
                epochs = num_epoch,
                verbose = 2,
                callbacks = [cp_callback,early_stopping,reduce_lr],
                validation_data = ( np.array(X_val),np.array(Y_val[:,np.newaxis]) ),
                class_weight = None
                )

        if plot_auc_curve:
            plt.figure(dpi=100)
            plt.plot(history.history['auc'])
            plt.plot(history.history['val_auc'])
            plt.legend(['training','validation'],loc='upper left')
            plt.title('auc')
            plt.savefig('model-training-auc-curve.png')
            plt.show()
        if plot_loss_curve:
            plt.figure(dpi=100)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.yscale('symlog')
            plt.legend(['training','validation'],loc='upper left')
            plt.title('loss')
            plt.savefig('model-training-loss-curve.png')
            plt.show()
    
   
    def run_RepeatedKFold(self,n_splits=3,n_repeats=5,
                          X_train=None,Y_train=None,
                          X_test=None,Y_test=None,
                          title=None,dir_result='./'
                         ):
        if X_train is None:
            X_train = self.Data['X_train']
        if Y_train is None:
            Y_train = self.Data['Y_train']
        if X_test is None:
            X_test = self.Data['X_test']
        if Y_test is None:
            Y_test = self.Data['Y_test']
            
        from sklearn.metrics import auc
        from sklearn.model_selection import RepeatedKFold
        rkf0 = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats, random_state=2652124)
        rkf1 = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats, random_state=1242652)
        index0 = np.array(np.where(Y_train==0)).squeeze()
        index1 = np.array(np.where(Y_train==1)).squeeze()
        #X0,X1,Y0,Y1 = X[index0,:],X[index1,:],Y[index0],Y[index1]
        X0,X1,Y0,Y1 = X_train[index0,:],X_train[index1,:],Y_train[index0],Y_train[index1]

#         import pdb
#         pdb.set_trace()
        tprs = []
        aucs = []
        thresholds = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(dpi=100)
        for (i0,(train_index0, val_index0)),(i1,(train_index1, val_index1)) in zip(enumerate(rkf0.split(Y0)),enumerate(rkf1.split(Y1)) ):
            
            X_train_cv = np.concatenate((X0[train_index0,:],X1[train_index1,:]),axis=0)
            X_val_cv = np.concatenate((X0[val_index0,:],X1[val_index1,:]),axis=0)
            Y_train_cv = np.concatenate((Y0[train_index0],Y1[train_index1]),axis=0)
            Y_val_cv = np.concatenate((Y0[val_index0],Y1[val_index1]),axis=0)
            # shuffle train-data
            arr = np.arange(np.shape(Y_train_cv)[0])
            np.random.shuffle(arr)
            X_train_cv = X_train_cv[arr,:]
            Y_train_cv = Y_train_cv[arr]
            
            self.model = DNN(self.input_node,self.output_node,self.hidden_layers_node,
                             optimizer=self.optimizer,
                             loss_function=self.loss_function,
                             L1_reg=self.L1_reg, L2_reg=self.L2_reg)
            
            self.train(X_train_cv, Y_train_cv,
                       X_val_cv, Y_val_cv,
                       num_epoch=self.max_epoch,
                       plot_loss_curve=False, plot_auc_curve=False)
#             import pdb
#             pdb.set_trace()
            viz = self.plot_roc_curve(X_val_cv, Y_val_cv,
                             label=None,    
                             name=None,#'ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)

            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            interp_threshold = np.interp(mean_fpr, viz.fpr, viz.threshold)
            interp_threshold[0] = 1.0
            thresholds.append(interp_threshold)
            aucs.append(viz.roc_auc)
#            print(i,aucs)
        #删除原来的legend
        ax.legend_.remove()#AttributeError: 'NoneType' object has no attribute 'remove'
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        mean_threshold = np.mean(thresholds, axis=0)
        mean_threshold[-1] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ci = stats.t.interval(alpha=0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs) )
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'micro-average ROC, AUC: %0.3f, 95%% CI: %0.3f~%0.3f' % (mean_auc, ci[0], ci[1]),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=title)
        ax.legend(loc="lower right")
        plt.savefig( os.path.join(dir_result,title.replace('\n','')) )
#        plt.show()
        ret = dict({})   
        ret['mean_fpr'],ret['mean_tpr'],ret['mean_threshold'] = mean_fpr, mean_tpr, mean_threshold
        ret['mean_auc'],ret['aucs'] = mean_auc, aucs
        dst = np.sqrt(ret['mean_fpr']**2+(ret['mean_tpr']-1)**2)
        best_threshold = ret['mean_threshold'][dst==dst.min()][0]
        self.best_threshold = best_threshold
        return ret,best_threshold,fig,ax

    def plot_roc_curve(self, X, Y,
                       label=None,
                       name=None,
                       alpha=0.3, lw=1, 
                       ax=None):
        from sklearn.metrics import roc_curve, RocCurveDisplay, auc
        Y_pred = self.predict_proba(X)
        fpr, tpr, threshold = roc_curve(Y, Y_pred, pos_label=None,
                                sample_weight=None,
                                drop_intermediate=True)
        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
        viz.threshold = threshold
        return viz.plot(ax=ax, name=name, alpha=alpha, lw=lw, label=label)
    
    def predict(self,X,threshold=0.5):
        self.model.load_weights(self.checkpoint_path)#restore the best model
        Y_pred = np.array(self.predict_proba(X)>threshold, dtype=np.int)
        return Y_pred
    def predict_proba(self,X):
        self.model.load_weights(self.checkpoint_path)#restore the best model
        Y_pred_prob = self.model.predict_proba(X).squeeze()
        return Y_pred_prob
    def test(
        self,
        X_train=None, Y_train=None,
        X_test=None, Y_test=None,
        max_epoch=None,
        class_names=None,
        best_threshold=None,
        dir_result='./'
    ):
        """
        测试集上的表现，输出roc曲线、混淆矩阵
        """
        from sklearn.metrics import confusion_matrix, classification_report
        from ultils_for_ML_binary_classify import plot_3cm, save_text
        if X_train is None:
            X_train = self.Data['X_train']
        if Y_train is None:
            Y_train = self.Data['Y_train']
        if X_test is None:
            X_test = self.Data['X_test']
        if Y_test is None:
            Y_test = self.Data['Y_test']
        if max_epoch is None:
            max_epoch = self.max_epoch

        if not os.path.isdir(dir_result):
            os.mkdir(dir_result)
        #(1)　测试集上的roc曲线
        self.train(X_train, Y_train,
                   X_test, Y_test,
                   num_epoch=max_epoch,
                   plot_loss_curve=False, 
                   plot_auc_curve=False)
        fig, ax = plt.subplots(dpi=100)
        title='ROC Curve of DNN against Test Set'
        fn =  os.path.join(dir_result,title.replace('\n','_'))+'.png'
        viz_test = self.plot_roc_curve(
            X_test, Y_test,
            label=None,
            name=None,#'ROC fold {}'.format(i),
            alpha=1, lw=2, ax=ax)
        
        roc_auc = viz_test.roc_auc
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
        ax.legend([r'ROC (AUC = %0.3f)' % (roc_auc),'Chance'],loc="lower right")
        plt.savefig( fn )
        plt.show()
        
        #(2) 画混淆矩阵
        if best_threshold:
            best_threshold = self.best_threshold
            y_pred = np.array(self.predict_proba(X_test)>best_threshold, dtype=np.int)
            #y_pred = np.argmax( np.array(self.predict_proba(X_test)>best_threshold, dtype=np.int), axis=1 )
        else:
            y_pred = np.array(self.predict(X_test), dtype=np.int)

        cm = confusion_matrix( y_true = Y_test, y_pred = y_pred )
        num_classes = len(class_names)
        plot_3cm(cm, class_names, num_classes, clf_name='DNN', dir_result=dir_result)
        
        #(3)获取report： specificity,precision,recall,f1-score等:
        # 1)获取原始report
        report = classification_report(
            y_true=Y_test, 
            y_pred=y_pred, 
            labels=np.unique(Y_test).tolist(),
            target_names=class_names,
            digits=3,
            output_dict=False
        )
        ## 2)补充roc_auc
        report = report+'\n'+'roc_auc: '+str(roc_auc.round(decimals=3))
        ## 3)保存
        fn = os.path.join(dir_result,'classification report of DNN against test set.txt')
        save_text(fn, report)
        print(report)
        
    #     # 画决策边界
    #     data = dict(X=X,Y=Y,classname=['no','yes'])
    #     visualize_reduced_decision_boundary(
    #         clf=classifier_fitted,data=data,
    #         title="Decision-Boundary of "+clf_name+" in TNSE-Projection",
    #         dir_result=dir_result
    #     ) 
        ret = dict( viz_test=viz_test, roc_auc=roc_auc, cm=cm, report=report)
        return ret


def DNN(input_node,output_node,hidden_layers_node,
        activation='relu',dropout_rate=0.2,
        optimizer=tf.keras.optimizers.Adam(),
        loss_function=tf.keras.losses.MeanAbsoluteError,
        L1_reg=0.002, L2_reg=0.002):
    '''
    create and compile DNN model
    '''
    
    model=tf.keras.Sequential()
    # hidden-layer_0
    model.add(Dense(units = hidden_layers_node[0],
                    input_shape=(input_node,),
                    activation = activation,
                    use_bias = True,
                    kernel_initializer='uniform',
                    kernel_regularizer = L1L2(l1=L1_reg,l2=L2_reg),
                    name = 'hidden-layer_0'
                    ))
    model.add(BatchNormalization(axis = -1))
#    model.add(Dropout(rate=dropout_rate, name='dropout-layer_0' ))
    # hidden layers
    for i in range(1,len(hidden_layers_node)):
        layer_name = 'hidden-layer_' + str(i+1)
        model.add(Dense(units = hidden_layers_node[i],
                        activation = activation,
                        use_bias = True,
                        kernel_initializer='random_normal',#'uniform',#,
                        kernel_regularizer=L1L2(l1=L1_reg,l2=L2_reg),
                        name = layer_name,
                        ))
        model.add(BatchNormalization(axis = -1))
        model.add(Dropout(rate=dropout_rate, name='dropout-layer_' + str(i+1)))  
    # Output of Network
    layer_name = 'output'
    model.add(Dense(units = output_node,
                    activation = 'sigmoid',#'softmax',
                    use_bias = True,
                    kernel_initializer='random_normal',#'uniform',
                    kernel_regularizer=L1L2(l1=L1_reg,l2=L2_reg),
                    name = layer_name
                    ))

    model.compile(optimizer=optimizer,
                  loss = loss_function,
                  metrics=[tf.keras.metrics.AUC(name='auc'),'accuracy'])
    return model



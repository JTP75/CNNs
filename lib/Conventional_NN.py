import numpy as np
import abc
if __name__ == "__main__":
    from utils.activations import sigmoid,dsigmoid
    from utils.perfmets import MSE
else:
    from lib.utils.activations import sigmoid,dsigmoid
    from lib.utils.perfmets import MSE


class layer(object):

    name:       str         # name
    size:       int         # input size
    out:        any         # output side
    din:        any         # input side
    weights:    any         # w
    dw:         any         # w change

    def __init__(self, name, size):
        self.name = name
        self.size = size

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def fprop(self, prev_layer: "layer"):
        pass

    @abc.abstractmethod
    def bprop_delta(self, next_layer: "layer"):
        pass

    @abc.abstractmethod
    def bprop_update(self, prev_layer: "layer", LR):
        pass


class FClayer(layer):

    out:        np.ndarray
    din:        np.ndarray
    weights:    np.ndarray
    dw:         np.ndarray

    def __init__(self,size,name="FCL_000"):

        super().__init__(name,size)                     # size = n
        self.out = None                                 # n+1,1
        self.din = None                                 # n+1,1
        self.weights = None                             # Nn,n+1
        self.dw = None                                  # Nn,n+1

    def __str__(self):
        
        return """
            \tname                    %s\n
            \tsize                    %d\n
            \tweights                   \n
        """ % (self.name,self.size) + np.array2string(self.weights) + "\n"

    def fprop(self,prev_layer):

        if len(prev_layer.out.shape) == 1:
            a = prev_layer.out.reshape((prev_layer.shape[0],1))
        else:
            a = prev_layer.out

        bias = np.ones((1,a.shape[1]))
        self.out = self.weights @ np.vstack([bias,a])

        return self.out
    
    def bprop_delta(self, next_layer: layer):                        ###########################

        self.din = (next_layer.weights[:,1:].T @ next_layer.din) * self.out*(1-self.out)        # df(z)
        return self.din
    
    def bprop_update(self, prev_layer: layer, LR):

        if len(prev_layer.out.shape)==1:
            a = np.reshape(prev_layer.out,(prev_layer.out.shape[0],1))
        else:
            a = prev_layer.out

        bias = np.ones((1,a.shape[1]))
        dW = -np.vstack((bias,a)) @ self.din.T
        self.weights -= LR * dW.T


class activationlayer(layer):

    out:        np.ndarray
    din:        np.ndarray
    weights:    np.ndarray
    dw:         np.ndarray
    __afunc:    function
    __dafunc:   function

    def __init__(self,size,name="ACL_000"):

        super().__init__(name,size)
        self.out = None                                 # n+1,1
        self.din = None                                 # n+1,1
        self.weights = None                             # Nn,n+1
        self.dw = None                                  # Nn,n+1
        self.set_activation()
    
    def set_activation(self, f: function=None, df: function=None):

        self.__afunc = f
        self.__dafunc = df

    def __str__(self):
        return self.__afunc.__name__()

    def fprop(self, prev_layer: layer):
        
        self.out = self.__afunc(prev_layer.out)
        return self.out

    def bprop_delta(self, next_layer: layer):
        
        self.din = self.__dafunc(next_layer.din)
        return self.din


class sigmoidlayer(activationlayer):
    
    def __init__(self, size, name="SiGL_000"):
        super().__init__(size, name)
        self.set_activation(sigmoid,dsigmoid)



class network(object):

    def __init__(self,*layer_sizes):

        self.layers = []
        self.perfmet = MSE
        for sz in layer_sizes:
            self.layers.append(FClayer(sz))

    def __str__(self):

        str = ""
        for i in range(len(self.layers)):
            str += "Layer %d:\n" % i
            str += self.layers[i].__str__()
        
    def init_random_weights(self):      #####

        for i in range(1,len(self.layers)):
            self.layers[i].weights = np.random.uniform(-1.0,1.0,(self.layers[i].size,self.layers[i-1].size+1))
        
    def fprop(self,batch):

        x = batch.T
        self.layers[0].out = x
        prev_layer = self.layers[0]
        for layer in self.layers[1:]: 
            x = layer.fprop(prev_layer)
            prev_layer = layer
        return x.T
    
    def bprop(self,batch,batch_resp,LR):

        self.layers[-1].din = (batch_resp - self.fprop(batch)).T

        next_layer = self.layers[-1]
        for layer in self.layers[-2::-1]:
            layer.bprop_delta(next_layer)
            next_layer = layer

        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.bprop_update(prev_layer,LR)
            prev_layer = layer

    def train(self,X_train,Y_train,LR,epochs,**kwargs):

        batch_size = 1
        print_freq = None
        gofast = False
        LR_fcn = lambda e: LR
        delta_perf_threshold = 1e-12
        shuffle = True
        check_convergence = False
        v = True

        for key,val in kwargs.items():
            if key=="batch_size":
                batch_size = val
            elif key=="print_freq":
                print_freq = val
            elif key=="fast":
                gofast = val
            elif key=="LR_fcn":
                LR_fcn = val
            elif key=="shuffle":
                shuffle = val
            elif key=="conv_delta":
                delta_perf_threshold = val
            elif key=="check_convergence":
                check_convergence = val
            elif key=="v":
                v = val

        full_perf = np.zeros((epochs,))
        batch_perf = np.zeros((epochs,))

        assert batch_size > 0
        if batch_size > X_train.shape[0]:
            batch_size = X_train.shape[0]

        data = np.hstack([X_train,Y_train])
        np.random.shuffle(data)

        print("\n============================================================") if v else None
        print("Beginning training routine...") if v else None
        
        epoch = 1
        set_idx = batch_size
        while epoch <= epochs:

            x = data[set_idx-batch_size:set_idx,:X_train.shape[1]]
            y = data[set_idx-batch_size:set_idx:,Y_train.shape[1]+1:]

            self.bprop(x,y,LR_fcn(epoch))

            set_idx += batch_size
            if set_idx > X_train.shape[0]:
                set_idx %= X_train.shape[0]
                if shuffle: 
                    np.random.shuffle(data)

                if not gofast:
                    Y_pred = self.fprop(X_train)
                    full_perf[epoch-1] = self.perfmet(Y_train,Y_pred)

                    yhat = self.fprop(x)
                    batch_perf[epoch-1] = self.perfmet(y,yhat)

                    if print_freq is not None and epoch%print_freq==0:
                        print("\033A    \033[A")
                        print("Epoch %d\tfull mse = %8.6f\tbatch mse = %8.6f" 
                            % (epoch, full_perf[epoch-1], batch_perf[epoch-1]), end="")
                        
                if v and check_convergence and epoch > 10 and np.abs(full_perf[epoch-2] - full_perf[epoch-1]) < delta_perf_threshold:
                
                    action = input("""  
                        Convergence declared after %d epochs What should the trainer do? > 
                    """ % epoch).split() if v else ["","",""]
                    if action[0] in ["return","exit","stop","terminate","end"]:
                        print("Training terminated.\n")
                        full_perf = np.nonzero(full_perf)
                        batch_perf = np.nonzero(batch_perf)
                        break
                    elif action[0] in ["continue","ignore","run","go"]:
                        print("Continuing...\n")
                        check_convergence = False
                    elif action[0] in ["set"]:
                        if action[1] in ["LR","a","alpha","learning_rate"]:
                            LR = np.float64(action[2])
                            print("New learning rate is %.6f\n" % LR)
                        elif action[1] in ["epochs","max_epochs","e","iters"]:
                            epochs = int(action[2])
                            print("New learning rate is %.6f\n" % LR)
                    elif action[0] in ["scale"]:
                        if action[1] in ["LR","a","alpha","learning_rate"]:
                            LR *= np.float64(action[2])
                            print("New learning rate is %.6f\n" % LR)
                    else:
                        print("Invalid keyword(s). Continuing...\n") if v else None
                        
                epoch += 1

        print("\nTraining Complete!") if v else None
        print("============================================================\n") if v else None
        if gofast:
            return 1,1
        else:
            return full_perf,batch_perf
        
    @property    
    def clear(self):
        self = network(layer.size for layer in self.layers)


if __name__ == "__main__":
    X_train = [[0,0],
               [1,0],
               [0,1],
               [1,1]]
    Y_train = [[0],
               [1],
               [1],
               [0]]
    
    x = np.array(X_train)
    y = np.array(Y_train)
    
    n = network(2,4,1)
    n.init_random_weights()
    fmse,bmse = n.train(x,y,0.01,10000,batch_size=2,print_freq=100)

    n.clear





import numpy as np
import abc
if __name__ == "__main__":
    from utils.activations import sigmoid,dsigmoid
    from utils.perfmets import MSE
    from utils.helpers
else:
    from lib.utils.activations import sigmoid,dsigmoid
    from lib.utils.perfmets import MSE


class layer(object):

    name:       str         # name
    size:       int         # input size
    OUT:        any         # OUTput side
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
    def initw(self):
        pass

    @abc.abstractmethod
    def fprop(self, prev: "layer") -> any:
        pass

    @abc.abstractmethod
    def bprop_delta(self, next: "layer") -> any:
        pass

    @abc.abstractmethod
    def bprop_update(self, LR) -> None:
        pass


class FClayer(layer):

    OUT:        np.ndarray
    din:        np.ndarray
    weights:    np.ndarray
    dw:         np.ndarray

    def __init__(self,size,name="FCL_000"): super().__init__(name,size)

    def __str__(self):
        
        return """
            \tname                    %s\n
            \tsize                    %d\n
            \tweights                   \n
        """ % (self.name,self.size) + np.array2string(self.weights) + "\n"
    
    def initw(self, next: layer):

        shape = (next.size,self.size)
        self.weights = np.random.uniform(-1.0,1.0,shape)

    def fprop(self, prev: layer):
            
        IN = prev.OUT
        self.OUT = self.weights @ IN
        return self.OUT
    
    def bprop_delta(self, next: layer):       ####

        dOUT = next.din
        self.din = self.weights.T @ dOUT        # inner prod
        self.dw = dOUT @ self.OUT.T             # outer prod
        return self.din
    
    def bprop_update(self, LR): self.weights -= LR * self.dw


class activationlayer(layer):

    OUT:        np.ndarray
    din:        np.ndarray
    weights:    None
    dw:         None
    __afunc:    any
    __dafunc:   any

    def __init__(self,size,name="ACL_000"):

        super().__init__(name,size)
        self.set_activation()
    
    def set_activation(self, f=None, df=None):

        self.__afunc = f
        self.__dafunc = df

    def __str__(self):

        return self.__afunc.__name__()

    def fprop(self, prev: layer):
        
        IN = prev.OUT
        self.OUT = self.__afunc(IN)
        return self.OUT

    def bprop_delta(self, next: layer):
        
        dOUT = next.din
        self.din = self.__dafunc(dOUT)
        return self.din


class sigmoidlayer(activationlayer):
    
    def __init__(self, size, name="SiGL_000"):
        super().__init__(size, name)
        self.set_activation(sigmoid,dsigmoid)


class inputlayer(layer):

    OUT:        np.ndarray
    din:        None
    weights:    None
    dw:         None
    shape:      tuple[int]
    

    def __init__(self, shape: tuple, name="INL_000"):

        self.shape = shape
        size = np.prod(self.shape)
        super().__init__(size, name)
        
    def fprop(self, prev: layer, network_input):

        if prev is None: assert network_input is not None
        self.OUT = network_input
        return self.OUT


class network(object):

    layers:     list[layer]     # list of MUTABLE layer sub-objects
    __perfmet:  any             # (function) network performance metric

    def __init__(self,*quick_FC_layers,**kwargs):

        self.layers = [inputlayer()]
        self.__perfmet = MSE
        for key,val in kwargs.items():
            if key in ["perf","perfmet"]:       self.__perfmet = val
            if key in ["load","ld","loadnet"]:  file = val

        for sz in quick_FC_layers:
            self.layers.append(FClayer(sz))

    def __str__(self):

        str = ""
        for i in range(len(self.layers)):
            str += "Layer %d:\n" % i
            str += self.layers[i].__str__()
        
    def init_random_weights(self):

        prev = None
        for curr in self.layers:
            if prev is not None:
                prev.initw(curr)
            prev = curr
        
    def fprop(self,batch) -> np.ndarray:

        prev = None
        for curr in self.layers:
            if type(curr)==inputlayer: curr.fprop(prev, batch)
            curr.fprop(prev)
            prev = curr
        return x.T
    
    def bprop(self, batch, actual_resp, LR):

        network_resp = self.fprop(batch)
        err0 = self.__perfmet(network_resp,actual_resp)
        
        next = None
        for curr in self.layers[::-1]:
            if next is None:
                curr.din = (network_resp - actual_resp)
            else:
                curr.bprop_delta(next)
            curr.bprop_update(LR)

        new_resp = self.fprop(batch)
        err1 = self.__perfmet(new_resp,actual_resp)

        return err1,err0

    def train(self,X_train,Y_train,LR,epochs,**kwargs):

        batch_size = 1
        print_freq = None
        LR_fcn = lambda e: LR
        derr_threshold = 1e-12
        shuffle = True
        check_convergence = False
        v = True

        for key,val in kwargs.items():
            if key=="batch_size":
                batch_size = val
            elif key=="print_freq":
                print_freq = val
            elif key=="LR_fcn":
                LR_fcn = val
            elif key=="shuffle":
                shuffle = val
            elif key=="derr_threshold":
                derr_threshold = val
            elif key=="check_convergence":
                check_convergence = val
            elif key=="v":
                v = val

        perf = np.zeros((epochs,))

        assert 0 < batch_size < X_train.shape[0]

        data = np.hstack([X_train,Y_train])
        np.random.shuffle(data)

        if v:
            print("\n============================================================")
            print("Beginning training routine...")
        
        epoch = 1
        set_idx = batch_size
        while epoch <= epochs:

            x = data[set_idx-batch_size:set_idx,:X_train.shape[1]]
            y = data[set_idx-batch_size:set_idx:,Y_train.shape[1]+1:]

            err1,err0 = self.bprop(x,y,LR_fcn(epoch))
            derr = np.abs(err1-err0)

            set_idx += batch_size
            if set_idx > X_train.shape[0]:
                set_idx %= X_train.shape[0]

                yhat = self.fprop(x)
                perf[epoch-1] = self.__perfmet(y,yhat)

                if print_freq is not None and epoch%print_freq==0 and v:
                    print("\033A    \033[A")
                    print("Epoch %d\tperformance = %8.6f (" + self.__perfmet.__name__ + ")\tdErr = %8.6f"
                          % (epoch, perf[epoch-1], derr), end="")
                        
                if v and check_convergence and derr < derr_threshold:
                
                    action = input("""  
                        Convergence declared after %d epochs What should the trainer do? > 
                    """ % epoch).split()
                    if action[0] in ["return","exit","stop","terminate","end"]:
                        print("Training terminated.\n")
                        perf = np.nonzero(perf)
                        batch_perf = np.nonzero(batch_perf)
                        break
                    elif action[0] in ["continue","ignore","run","go"]:
                        print("Continuing...\n")
                        check_convergence = False
                    elif action[0] in ["set"]:
                        if action[1] in ["LR","a","alpha","learning_rate","lr"]:
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
                        print("Invalid keyword(s). Continuing...\n")
                        
                epoch += 1
                if shuffle: np.random.shuffle(data)
        if v:
            print("\nTraining Complete!")
            print("============================================================\n")
        return perf
        


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

    print(n)



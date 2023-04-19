import numpy as np
import abc
if __name__ == "__main__":
    from utils.activations import sigmoid,dsigmoid
    from utils.perfmets import MSE
    from utils.helpers import *
else:
    from lib.utils.activations import sigmoid,dsigmoid
    from lib.utils.perfmets import MSE
    from lib.utils.helpers import *

np.seterr("raise")

class layer(object):

    name:       str         # name
    inshape:    tuple[int]  # input size
    outshape:   tuple[int]  # output size
    bias:       bool        # bias
    OUT:        any         # OUTput side
    dIN:        any         # input side
    w:          any         # w
    dw:         any         # w change

    def __init__(self, inshape, outshape=None, bias=False, name=None):

        if type(inshape)==int: inshape = (inshape,1)
        if type(outshape)==int: outshape = (outshape,1)
        self.name = name
        self.inshape = inshape
        self.outshape = outshape
        self.bias = bias

        if self.outshape is None: 
            self.outshape = self.inshape

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def initw(self) -> None:
        pass

    @abc.abstractmethod
    def check_shapes(self, prev: "layer", next: "layer") -> None:
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

    @property
    @abc.abstractmethod
    def has_learnable_params(self) -> bool:
        pass

class FClayer(layer):

    OUT:        np.ndarray
    dIN:        np.ndarray
    w:          np.ndarray
    dw:         np.ndarray

    def __init__(self,inshape,outshape=None,bias=True,name="FC_layer_0"): 
        
        super().__init__(inshape,outshape,bias,name)
        if self.bias:   wshape = (self.outshape[0],self.inshape[0]+1)
        else:           wshape = (self.outshape[0],self.inshape[0])
        self.OUT = np.empty(self.outshape,dtype=np.float64)
        self.dIN = np.empty(self.inshape,dtype=np.float64)
        self.w = np.empty(wshape,dtype=np.float64)
        self.dw = np.empty(wshape,dtype=np.float64)

    def __str__(self):
        
        return """
            \tname                    %s\n
            \tsize                    %d\n
            \tweights                   \n
        """ % (self.name,self.size) + np.array2string(self.w) + "\n"
    
    def check_shapes(self, prev:layer, next:layer, v=False) -> None:
        
        # fprop shapes
        IN = prev.OUT.shape
        THETA = self.w.shape
        OUT = self.OUT.shape
        # bprop
        dIN = self.dIN.shape
        dTHETA = self.dw.shape
        dOUT = next.dIN.shape

        # verbosity (dbg)
        if v:
            print("\nFCLayer:\t", OUT, "\t=\t", THETA, "\t@\t", IN)
            print("FCLayer deltas:\t", dTHETA, "\t=\t", dOUT, "\t@\t", dIN[::-1])

        # check shape deltas
        assert IN == dIN, "Assertion failed: IN != dIN"
        assert OUT == dOUT, "Assertion failed: OUT != dOUT"
        assert THETA == dTHETA, "Assertion failed: THETA != dTHETA"
        # check num samples
        assert IN[1] == OUT[1] == dIN[1] == dOUT[1], "Assertion failed: nsamples"
        # check shape transitions:
        assert IN[0] == THETA[1], "Assertion failed: IN[0] != THETA[1]"
        assert OUT[0] == THETA[0], "Assertion failed: OUT[0] != THETA[0]"
        assert dIN[0] == dTHETA[1], "Assertion failed: dIN[0] != dTHETA[1]"
        assert dOUT[0] == dTHETA[0], "Assertion failed: dOUT[0] != dTHETA[0]"
    
    def initw(self):

        self.w = np.random.uniform(-1.0,1.0,self.w.shape)

    def fprop(self, prev: layer):
            
        IN: np.ndarray = prev.OUT
        if self.bias:
            B_vect = np.ones((1,IN.shape[1]))
            IN = np.vstack((B_vect,IN))
        try:
            self.OUT = self.w @ IN
        except FloatingPointError:
            raise RuntimeError
        return self.OUT
    
    def bprop_delta(self, next: layer):
        
        dOUT = next.dIN
        assert dOUT.shape == self.OUT.shape, "dOUT and OUT shapes mismatch."
        try:
            self.dIN = (self.w.T @ dOUT)
            self.dw = self.OUT @ self.dIN.T
        except FloatingPointError:
            raise RuntimeError
        self.dIN = self.dIN[1:]
        assert self.dw.shape == self.w.shape, "dw and w shapes mismatch."
        return self.dIN
    
    def bprop_update(self, LR): 
        
        self.w -= LR * self.dw

    @property
    def has_learnable_params(self) -> bool: 
        return True

class activationlayer(layer):

    OUT:        np.ndarray
    dIN:        np.ndarray
    w:          None
    dw:         None
    __afunc:    any
    __dafunc:   any

    def __init__(self,shape,name="activation_layer_0"):

        super().__init__(shape,shape,False,name)
        self.OUT = np.empty(self.outshape,dtype=np.float64)
        self.dIN = np.empty(self.outshape,dtype=np.float64)
    
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
        
        dOUT = next.dIN
        self.dIN = self.__dafunc(dOUT)
        return self.dIN
    
    @property
    def has_learnable_params(self) -> bool: 
        return False
    
class sigmoidlayer(activationlayer):
    
    def __init__(self, shape, name="sigmoid_layer_0"):
        super().__init__(shape, name)
        self.set_activation(sigmoid,dsigmoid)

class inputlayer(layer):

    OUT:        np.ndarray
    dIN:        None
    w:          None
    dw:         None
    nsamples:   int

    def __init__(self, shape, name="input_layer_0"):

        super().__init__(0, shape, False, name)
        self.OUT = np.empty(self.outshape)
        self.nsamples = 0
        
    def fprop(self, prev: layer, network_input):

        prev = None
        assert type(network_input) == np.ndarray, "network_input is not numpy.ndarray"
        if network_input.shape[0] != self.outshape[0]: network_input = network_input.T
        assert network_input.shape[0] == self.outshape[0], f"network_input for N samples must be of shape ({self.outshape[0]},N)"
        self.nsamples = network_input.shape[1]

        self.OUT = network_input
        return self.OUT

    @property
    def has_learnable_params(self) -> bool: 
        return False

class network:

    layers:     list[layer]     # list of MUTABLE layer sub-objects
    __perfmet:  any             # (function) network performance metric
    in_shape:   tuple
    out_shape:  tuple

    def __init__(self, *quick_FC_layers: int, **kwargs):

        self.layers = []
        self.__perfmet = MSE
        for key,val in kwargs.items():
            if key in ["perf","perfmet"]:       self.__perfmet = val
            if key in ["load","ld","loadnet"]:  file = val

        self.in_shape = (quick_FC_layers[0],1)
        self.out_shape = (quick_FC_layers[-1],1)

        prev_size = self.in_shape[0]
        for idx,size in enumerate(quick_FC_layers[1:]): 
            self.layers.append(FClayer(prev_size,size,name=f"FC_Layer_{idx}"))
            self.layers.append(sigmoidlayer(size,name=f"Sigmoid_Layer_{idx}"))
            prev_size=size
        self.layers.insert(0,inputlayer(self.in_shape[0]))

        return

    def __str__(self):

        str = ""
        for i,layer in enumerate(self.layers):
            str += "Layer %d:\n" % i
            str += layer.__str__()

    def check_shapes(self,v=False):
        for i in range(1,len(self.layers)-1):
            prev, curr, next = self.layers[i-1], self.layers[i], self.layers[i+1]
            curr.check_shapes(prev,next,v)

        
    def init_random_weights(self):

        prev = None
        for curr in self.layers:
            if prev is not None:
                prev.initw()
            prev = curr
        
    def fprop(self,batch) -> np.ndarray:

        N = batch.shape[0]

        prev = None
        for curr in self.layers:
            if prev is None:
                curr.fprop(prev, batch)
                assert curr.nsamples==N
            else: 
                network_resp = curr.fprop(prev)
            assert curr.OUT.shape[1]==N
            prev = curr
        return network_resp
    
    def bprop(self, batch, actual_resp, LR):

        network_resp = self.fprop(batch)
        err0 = self.__perfmet(network_resp, actual_resp.T)
        
        next = None
        for curr in self.layers[::-1]:
            if next is None:
                curr.dIN = (network_resp - actual_resp.T)
            else:
                curr.bprop_delta(next)
                curr.bprop_update(LR)
            next = curr

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
        ncpus = 1

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
            elif key=="ncpus":
                ncpus = val

        assert 0 < batch_size <= X_train.shape[0]

        data = np.hstack([X_train,Y_train])
        np.random.shuffle(data)
        perf = np.zeros((epochs,))
        if v:
            print("\n============================================================")
            print("Beginning training routine...")
        
        epoch = 1
        set_idx = 0
        while epoch <= epochs:

            x = data[set_idx:set_idx+batch_size, :X_train.shape[1]]
            y = data[set_idx:set_idx+batch_size, X_train.shape[1]:]

            alpha = LR_fcn(epoch)
            pfunc = lambda x,y: self.bprop(x,y,alpha)

            """
            xp = np.array_split(x,ncpus)
            yp = np.array_split(y,ncpus)
            pdata = [(xp[i],yp[i]) for i in range(batch_size)]

            with Pool(ncpus) as pool:
                err1,err0 = pool.map(pfunc,pdata)
                pool.close()
                pool.join()
            """

            err1,err0 = self.bprop(x,y,alpha)
            derr = err1-err0

            set_idx += batch_size
            if set_idx >= X_train.shape[0]:
                set_idx %= X_train.shape[0]

                yhat = self.fprop(x)
                perf[epoch-1] = self.__perfmet(y,yhat)

                if v and print_freq is not None and epoch%print_freq==0:
                    err = self.__perfmet(Y_train, self.fprop(X_train))
                    print("\033A    \033[A")
                    print(f"Epoch {epoch}\tperformance = {perf[epoch-1]}\tdErr = {derr}", end="")
                        
                epoch += 1

                if shuffle:
                    np.random.shuffle(data)
        if v:
            print("\nTraining Complete!")
            print("============================================================\n")
        return perf
    
    def print(self):
        for layer in self.layers:
            print(layer.name)

        
if __name__ == "__main__":

    X_train = [[0,0],
               [1,0],
               [0,1],
               [1,1]]
    Y_train = [[0,1],
               [1,0],
               [1,0],
               [0,0]]
    
    x = np.array(X_train)
    y = np.array(Y_train)
    
    n = network(2,6,2)

    n.print()

    exit()
    n.init_random_weights()
    n.train(x,y,0.1,10000,batch_size=4,print_freq=10)

    resp = n.fprop(x)
    print(resp)
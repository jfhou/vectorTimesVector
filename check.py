
import theano
from theano import gof

class VectorTimesVectorGrad(gof.COp):
    __props__ = ()

    func_file = "./vectorTimesVectorGrad.c"
    func_name = "APPLY_SPECIFIC(vector_times_vector_grad)"


    def __init__(self):
        super(VectorTimesVectorGrad, self).__init__(self.func_file,
                                                self.func_name)

    def make_node(self, x, y, gz):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 1:
            raise TypeError('y must be a 1-d vector')

        # Create an output variable of the same type as x
        gx = theano.tensor.TensorType(
                        dtype=theano.scalar.upcast(x.dtype, y.dtype),
                        broadcastable=[False])()
        gy = theano.tensor.TensorType(
                        dtype=theano.scalar.upcast(x.dtype, y.dtype),
                        broadcastable=[False])()


        return gof.Apply(self, [x, y, gz], [gx, gy])
        
vector_times_vector_grad=VectorTimesVectorGrad()


class VectorTimesVector(gof.COp):
    __props__ = ()

    func_file = "./vectorTimesVector.c"
    func_name = "APPLY_SPECIFIC(vector_times_vector)"


    def __init__(self):
        super(VectorTimesVector, self).__init__(self.func_file,
                                                self.func_name)

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 1:
            raise TypeError('y must be a 1-d vector')

        # Create an output variable of the same type as x
        output_var = theano.tensor.TensorType(
                        dtype=theano.scalar.upcast(x.dtype, y.dtype),
                        broadcastable=[False])()

        return gof.Apply(self, [x, y], [output_var])
    
    
    def grad(self, inputs, gradients):
        x,y=inputs
        gz,=gradients
        return vector_times_vector_grad(x,y,gz)
        
        
vector_times_vector=VectorTimesVector()




import numpy
from theano import tensor
import scipy
from scipy import io
a=tensor.vector('a',dtype='float32')
b=tensor.vector('b',dtype='float32')
c=vector_times_vector(a,b)
f=theano.function([a,b],c)

ga,gb=theano.grad(c.sum(),[a,b])
g=theano.function([a,b],[ga,gb])


x=numpy.random.randn(4).astype('float32')
y=numpy.random.randn(4).astype('float32')
z=f(x,y)
print 'x'
print x
print 'y'
print y
print 'z'
print z

gx,gy=g(x,y)
print 'gx'
print gx
print 'gy'
print gy

#scipy.io.savemat('check_fsmn.mat',{'x':x,'y':y,'z':z})
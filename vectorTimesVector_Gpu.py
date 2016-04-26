
import scipy
from scipy import io
import theano
from theano import gof
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, host_from_gpu
from theano import Op, Apply
#class VectorTimesVectorGrad(GpuOp):
#    __props__ = ()
#
#    func_file = "./vectorTimesVectorGrad.cu"
#    func_name = "APPLY_SPECIFIC(vector_times_vector_grad)"
#
#
#    def __init__(self):
#        super(VectorTimesVectorGrad, self).__init__(self.func_file,
#                                                self.func_name)
#
#    def make_node(self, x, y, gz):
#        x = as_cuda_ndarray_variable(x)
#        y = as_cuda_ndarray_variable(y)
#        gz = as_cuda_ndarray_variable(gz)
#        gx = x.type()
#        gy = y.type()
#
#        return Apply(self, [x, y, gz], [gx, gy])
#        
#vector_times_vector_grad=VectorTimesVectorGrad()


class VectorTimesVector(GpuOp):
    
    
    nin=2
    nout=1
#    def __init__(self, **kwargs):
#        Op.__init__(self, **kwargs)

    def make_node(self, x, y):
        x = as_cuda_ndarray_variable(x)
        y = as_cuda_ndarray_variable(y)
        z = x.type()
        return Apply(self, [x, y], [z])
    
    def c_support_code(self):
        return '''
            __global__ void vector_elemwise_mult(
            const float * x_ptr, int x_str,
            const float * y_ptr, int y_str,
            float * z_ptr, int z_str, int nbElements)
            {
                int idx=blockIdx.x*blockDim.x + threadIdx.x;
                if(idx<nbElements)
                {
                    z_ptr[idx * z_str] = x_ptr[idx * x_str] * y_ptr[idx * y_str];
                }
            }
        '''
    
    def c_code(self, node, nodename, inp, out, sub):
        input0,input1=inp
        output0,=out
        fail = sub['fail']
        return '''
        
        
        
        
        
            // Validate that the inputs have the same shape
    if ( !(CudaNdarray_HOST_DIMS(%(input0)s)[0] == CudaNdarray_HOST_DIMS(%(input1)s)[0]))
    {
        
        %(fail)s;
    }

    // Validate that the output storage exists and has the same
    // dimension as x.
    if (NULL == (%(output0)s) || !(CudaNdarray_HOST_DIMS(%(input0)s)[0] == CudaNdarray_HOST_DIMS(%(output0)s)[0]))
    {
        /* Reference received to invalid output variable.
        Decrease received reference's ref count and allocate new
        output variable */
        Py_XDECREF((%(output0)s));
        (%(output0)s) = (CudaNdarray*)CudaNdarray_NewDims(1,CudaNdarray_HOST_DIMS(%(input0)s));

        if (!(%(output0)s)) {

            %(fail)s;
        }
    }

    {
    // kernel function
    int n_blocks = (int)(CudaNdarray_HOST_DIMS(%(input0)s)[0]/256)+1;
    int n_threads = std::min(CudaNdarray_HOST_DIMS(%(input0)s)[0],256);
    //int n_shared_bytes = n_threads * sizeof(float);
    int n_shared_bytes = 0;
    vector_elemwise_mult<<<n_blocks, n_threads, n_shared_bytes>>>(CudaNdarray_DEV_DATA(%(input0)s),CudaNdarray_HOST_STRIDES(%(input0)s)[0],
    CudaNdarray_DEV_DATA(%(input1)s),CudaNdarray_HOST_STRIDES(%(input1)s)[0],
    CudaNdarray_DEV_DATA((%(output0)s)),CudaNdarray_HOST_STRIDES((%(output0)s))[0],CudaNdarray_HOST_DIMS(%(input0)s)[0]);
    }

        
        
        ''' % locals()
        
        
        
    
#    def grad(self, inputs, gradients):
#        x,y=inputs
#        gz,=gradients
#        return vector_times_vector_grad(x,y,gz)
        
        
vector_times_vector=VectorTimesVector()




import numpy
from theano import tensor
import scipy
from scipy import io
a=tensor.vector('a',dtype='float32')
b=tensor.vector('b',dtype='float32')
c=vector_times_vector(a,b)
f=theano.function([a,b],host_from_gpu(c))

#ga,gb=theano.grad(c.sum(),[a,b])
#g=theano.function([a,b],[ga,gb])


x=numpy.random.randn(1000).astype('float32')
y=numpy.random.randn(1000).astype('float32')
z=f(x,y)
print 'x'
print x
print 'y'
print y
print 'z'
print z

#gx,gy=g(x,y)
#print 'gx'
#print gx
#print 'gy'
#print gy

scipy.io.savemat('check_fsmn.mat',{'x':x,'y':y,'z':z})
import megengine.module as M
import megengine.functional as F
import numpy as np

if __name__ == '__main__':

   import megengine.hub
   import megengine.functional as F
   from megengine.jit import trace

   net = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
   net.eval()

   @trace(symbolic=True)
   def fun(data,*, net):
      pred = net(data)
      pred_normalized = F.softmax(pred)
      return pred_normalized

   data = np.random.random([1, 3, 224,
                           224]).astype(np.float32)


   fun.trace(data,net=net)
   fun.dump("shufflenet_deploy.mge", arg_names=["data"], optimize_for_inference=True)
   
-- resnet model code derived from: https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
local SpatialConvolution = cudnn.SpatialConvolution or nn.SpatialConvolution
local SpatialFullConvolution =cudnn.SpatialFullConvolution or nn.SpatialFullConvolution
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization  or nn.SpatialBatchNormalization 
local SpatialMaxPooling = nn.SpatialMaxPooling
local ReLU = nn.ReLU
local LeakyReLU = nn.LeakyReLU
local ndf = opt.ndf or 64
local ngf = opt.ngf or 64
local nc = opt.geometry[1]

local function add_dcgan_layer(net, nin, nout)
  net:add(SpatialConvolution(nin, nout, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(nout)):add(LeakyReLU(0.2, true))
end

local function add_dcgan_full_layer(net, nin, nout)
  net:add(SpatialFullConvolution(nin, nout, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(nout)):add(LeakyReLU(0.2, true))
end

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
  local useConv = shortcutType == 'C' or
     (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
     -- 1x1 convolution
     return nn.Sequential()
        :add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        :add(SpatialBatchNormalization(nOutputPlane))
  elseif nInputPlane ~= nOutputPlane then
     -- Strided, zero-padded identity shortcut
     return nn.Sequential()
        :add(nn.SpatialAveragePooling(1, 1, stride, stride))
        :add(nn.Concat(2)
           :add(nn.Identity())
           :add(nn.MulConstant(0)))
  else
     return nn.Identity()
  end
end

local function upsample_shortcut(nInputPlane, nOutputPlane, stride)
  local useConv = shortcutType == 'C' or
     (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
    assert(false)
     -- 1x1 convolution
     return nn.Sequential()
        :add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        :add(SpatialBatchNormalization(nOutputPlane))
  elseif nInputPlane ~= nOutputPlane then
     -- Strided, zero-padded identity shortcut
     local s = nn.Sequential()
     if stride == 2 then 
       s:add(nn.SpatialUpsamplingNearest(2))
     end
     s:add(nn.Concat(2)
           :add(nn.Identity())
           :add(nn.MulConstant(0)))
    return s
  else
     return nn.Identity()
  end
end

-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
local function basicblock(n, stride)
  local nInputPlane = iChannels
  iChannels = n

  local s = nn.Sequential()
  s:add(SpatialConvolution(nInputPlane,n,3,3,stride,stride,1,1))
  s:add(SpatialBatchNormalization(n))
  s:add(ReLU(true))
  s:add(SpatialConvolution(n,n,3,3,1,1,1,1))
  s:add(SpatialBatchNormalization(n))

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(true))
end

local function upsampleblock(n, stride)
  local nInputPlane = iChannels
  iChannels = n

  local s = nn.Sequential()
  if stride == 1 then
    s:add(SpatialConvolution(nInputPlane,n,3,3,stride,stride,1,1))
  else
    s:add(SpatialFullConvolution(nInputPlane,n,4, 4, 2, 2, 1, 1))
  end
  s:add(SpatialBatchNormalization(n))
  s:add(ReLU(true))
  s:add(SpatialConvolution(n,n,3,3,1,1,1,1))
  s:add(SpatialBatchNormalization(n))

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(upsample_shortcut(nInputPlane, n, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(true))
end

-- The bottleneck residual layer for 50, 101, and 152 layer networks
local function bottleneck(n, stride)
  local nInputPlane = iChannels
  iChannels = n * 4

  local s = nn.Sequential()
  s:add(SpatialConvolution(nInputPlane,n,1,1,1,1,0,0))
  s:add(SpatialBatchNormalization(n))
  s:add(ReLU(true))
  s:add(SpatialConvolution(n,n,3,3,stride,stride,1,1))
  s:add(SpatialBatchNormalization(n))
  s:add(ReLU(true))
  s:add(SpatialConvolution(n,n*4,1,1,1,1,0,0))
  s:add(SpatialBatchNormalization(n * 4))

  return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n * 4, stride)))
     :add(nn.CAddTable(true))
     :add(ReLU(true))
end

-- Creates count residual blocks with specified number of features
local function layer(block, features, count, stride)
  local s = nn.Sequential()
  for i=1,count do
     s:add(block(features, i == 1 and stride or 1))
  end
  return s
end

function makeContentEncoder()
  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
 local depth = opt.depth or 18
  local cfg = {
     [18]  = {{2, 2, 2, 2}, 512, basicblock},
     [34]  = {{3, 4, 6, 3}, 512, basicblock},
  }

  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures, block = table.unpack(cfg[depth])
  iChannels = 64
  print(' | ResNet-' .. depth .. ' ImageNet')

  -- The ResNet ImageNet model
  local net = nn.Sequential()
  net:add(nn.JoinTable(2))
  net:add(SpatialConvolution(nc*opt.nShare,64,5,5,2,2,3,3))
  net:add(SpatialBatchNormalization(64))
  net:add(ReLU(true))
  net:add(SpatialMaxPooling(3,3,2,2,1,1))
  net:add(layer(block, 64, def[1]))
  net:add(layer(block, 128, def[2], 2))
  net:add(layer(block, 256, def[3], 2))
  net:add(layer(block, 512, def[4], 2))
  --net:add(nn.SpatialDropout())
  net:add(SpatialConvolution(nFeatures, opt.latentDim, 3, 3))
  model:add(SpatialBatchNormalization(opt.latentDim)):add(nn.Tanh())

  if opt.normalize then
    net:add(nn.Reshape(opt.latentDim))
    net:add(nn.Normalize(2))
    net:add(nn.Reshape(opt.latentDim, 1, 1))
  end
  initModel(net)
  return net
end

local function res_lrelu_block(h, nf)
  local conv1 = SpatialConvolution(nf, nf, 3, 3, 1, 1, 1, 1)(h) 
  local act1 = LeakyReLU(0.2, true)(SpatialBatchNormalization(nf)(conv1))
  local conv2 = SpatialConvolution(nf, nf, 3, 3, 1, 1, 1, 1)(act1) 
  local act2 = LeakyReLU(0.2, true)(SpatialBatchNormalization(nf)(conv2))
  return nn.CAddTable()({h, act2})
end

function makeResnetDecoder()
  local latent  = nn.Identity()()
  local action  = nn.Identity()()
  local inp = SpatialFullConvolution(opt.latentDim+opt.actionDim, 512, 4, 4)(nn.JoinTable(2)({latent, action}))
  local act = LeakyReLU(0.2, true)(SpatialBatchNormalization(512)(inp))
  -- 512x4x4
  local block1 = res_lrelu_block(act, 512) 
  local up1 = LeakyReLU(0.2, true)(SpatialBatchNormalization(256)(SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1)(block1) ))
  -- 256x8x8
  local block2 = res_lrelu_block(up1, 256) 
  local block3 = res_lrelu_block(block2, 256) 
  local up3 = LeakyReLU(0.2, true)(SpatialBatchNormalization(128)(SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1)(block3) ))
  -- 128x16x16
  local block4 = res_lrelu_block(up3, 128)
  local block5 = res_lrelu_block(block4, 128)
  local up5 = LeakyReLU(0.2, true)(SpatialBatchNormalization(64)(SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1)(block5) ))
  -- 64x32x32 
  local block6 = res_lrelu_block(up5, 64)
  local block7 = res_lrelu_block(block6, 64)
  local block8 = res_lrelu_block(block7, 64)
  local up8 = SpatialFullConvolution(64, nc, 4, 4, 2, 2, 1, 1)(block8)
  -- ncx64x64 
  local out 
  if opt.output == 'sigmoid' then
    out = nn.Sigmoid()(up8)
  elseif opt.output == 'tanh' then
    out = nn.Tanh()(up8)
  else
    assert(false)
  end
  local net = nn.gModule({latent, action}, {out})
  
  initModel(net)
  return net
end

function makeJointDecoder()
  if opt.decoder == 'resnet' then
    return makeResnetDecoder()
  else
    return makeDCGANDecoder()
  end
end

function makeDCGANDecoder()
  -- predictor 
  local net = nn.Sequential()
  net:add(nn.JoinTable(2))
  net:add(SpatialFullConvolution(opt.actionDim+opt.latentDim, ngf*8, 4, 4))
  net:add(SpatialBatchNormalization(ngf*8)):add(nn.LeakyReLU(0.2, true))
  add_dcgan_full_layer(net, ngf*8, ngf*4) -- 2 -> 4
  add_dcgan_full_layer(net, ngf*4, ngf*2) -- 4 -> 8
  add_dcgan_full_layer(net, ngf*2, ngf) -- 8 -> 16 
  net:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
  net:add(nn.Sigmoid())
 return net
end

function makePoseEncoder()
  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
 local depth = opt.depth or 18
  local cfg = {
     [18]  = {{2, 2, 2, 2}, 512, basicblock},
     [34]  = {{3, 4, 6, 3}, 512, basicblock},
  }

  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures, block = table.unpack(cfg[depth])
  iChannels = 64
  print(' | ResNet-' .. depth .. ' ImageNet')

  -- The ResNet ImageNet model
  local net = nn.Sequential()
  net:add(SpatialConvolution(nc,64,5,5,2,2,3,3))
  net:add(SpatialBatchNormalization(64))
  net:add(ReLU(true))
  net:add(SpatialMaxPooling(3,3,2,2,1,1))
  net:add(layer(block, 64, def[1]))
  net:add(layer(block, 128, def[2], 2))
  net:add(layer(block, 256, def[3], 2))
  net:add(layer(block, 512, def[4], 2))
  net:add(SpatialConvolution(nFeatures, opt.actionDim, 3, 3))
  net:add(SpatialBatchNormalization(opt.actionDim)):add(nn.Tanh())

  if opt.normalize then
    net:add(nn.Reshape(opt.actionDim))
    net:add(nn.Normalize(2))
    net:add(nn.Reshape(opt.actionDim, 1, 1))
  end
  initModel(net)
  return net
end

function makeSceneDiscriminator()
  local nf = 100
  local net = nn.Sequential()
  net:add(nn.JoinTable(2))
  net:add(nn.Reshape(opt.poseDim*2))
  net:add(nn.Linear(opt.poseDim*2, nf))
  net:add(nn.ReLU())
  net:add(nn.Linear(nf, nf))
  net:add(nn.ReLU())
  net:add(nn.Linear(nf, 1))
  net:add(nn.Sigmoid())
  initModel(net)
  return net 

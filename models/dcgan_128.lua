local SpatialConvolution = cudnn.SpatialConvolution or nn.SpatialConvolution
local SpatialFullConvolution =cudnn.SpatialFullConvolution or nn.SpatialFullConvolution
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization  or nn.SpatialBatchNormalization 
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

function makeContentEncoder()
  local net = nn.Sequential()
  net:add(nn.JoinTable(2))
  add_dcgan_layer(net, nc*opt.nShare, ngf) -- 128 -> 64
  add_dcgan_layer(net, ngf, ngf*2) -- 64 -> 32
  add_dcgan_layer(net, ngf*2, ngf*4) -- 32 -> 16 
  add_dcgan_layer(net, ngf*4, ngf*8) -- 16 -> 8
  add_dcgan_layer(net, ngf*8, ngf*8) -- 8 -> 4
  net:add(SpatialConvolution(ngf*8, opt.contentDim, 4, 4))
  net:add(SpatialBatchNormalization(opt.contentDim)):add(nn.Tanh())--:add(LeakyReLU(0.2, true))
  if opt.normalize then
    net:add(nn.Reshape(opt.contentDim))
    net:add(nn.Normalize(2))
    net:add(nn.Reshape(opt.contentDim, 1, 1))
  end
  initModel(net)
  return net
end

function makePoseEncoder()
  local net = nn.Sequential()
  add_dcgan_layer(net, nc, ngf) -- 128 -> 64
  add_dcgan_layer(net, ngf, ngf*2) -- 64 -> 32 
  add_dcgan_layer(net, ngf*2, ngf*4) -- 32 -> 16 
  add_dcgan_layer(net, ngf*4, ngf*8) -- 16 -> 8
  add_dcgan_layer(net, ngf*8, ngf*8) -- 8 -> 4
  net:add(SpatialConvolution(ngf*8, opt.poseDim, 4, 4))
  net:add(SpatialBatchNormalization(opt.poseDim)):add(nn.Tanh()) --:add(LeakyReLU(0.2, true))
  if opt.normalize then
    net:add(nn.Reshape(opt.poseDim))
    net:add(nn.Normalize(2))
    net:add(nn.Reshape(opt.poseDim, 1, 1))
  end
  initModel(net)
  return net
end

function makeDecoder()
  local net = nn.Sequential()
  net:add(nn.JoinTable(2))
  net:add(SpatialFullConvolution(opt.poseDim+opt.contentDim, ngf*8, 4, 4)) -- 1 -> 4
  net:add(SpatialBatchNormalization(ngf*8)):add(nn.LeakyReLU(0.2, true))
  add_dcgan_full_layer(net, ngf*8, ngf*8) --  4 -> 8
  add_dcgan_full_layer(net, ngf*8, ngf*4) --  8 -> 16
  add_dcgan_full_layer(net, ngf*4, ngf*2) --  16 -> 32
  add_dcgan_full_layer(net, ngf*2, ngf) -- 32 -> 64
  net:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)) -- 64 -> 128
  net:add(nn.Sigmoid())
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
end

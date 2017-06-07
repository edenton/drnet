local SpatialConvolution = cudnn.SpatialConvolution or nn.SpatialConvolution
local SpatialFullConvolution =cudnn.SpatialFullConvolution or nn.SpatialFullConvolution
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization  or nn.SpatialBatchNormalization 
local SpatialMaxPooling = cudnn.SpatialMaxPooling  or nn.SpatialMaxPooling 
local ReLU = nn.ReLU
local SpatialUpSamplingNearest = nn.SpatialUpSamplingNearest
local LeakyReLU = nn.LeakyReLU
local ndf = opt.ndf or 64
local ngf = opt.ngf or 64
local nc = opt.geometry[1]

function makeUnetDCGAN()
  -- shared encoder
  local x = nn.Identity()()
  local pose = nn.Identity()()

  -- nc*opt.nShare x 128 x 128
  local enc1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf)(SpatialConvolution(nc*opt.nShare, ngf, 4, 4, 2, 2, 1, 1)(nn.JoinTable(2)(x))))
  -- ngf x 64 x 64
  local enc2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*2)(SpatialConvolution(ngf, ngf*2, 4, 4, 2, 2, 1, 1)(enc1)))
  -- ngf*2 x 32 x 32 
  local enc3 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*4)(SpatialConvolution(ngf*2, ngf*4, 4, 4, 2, 2, 1, 1)(enc2)))
  -- ngf*4 x 16 x 16 
  local enc4 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*8)(SpatialConvolution(ngf*4, ngf*8, 4, 4, 2, 2, 1, 1)(enc3)))
  -- ngf*8 x 8 x 8 
  local enc5 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*8)(SpatialConvolution(ngf*8, ngf*8, 4, 4, 2, 2, 1, 1)(enc4)))
  -- ngf*8 x 4 x 4 
  local enc6 = nn.Tanh()(SpatialBatchNormalization(opt.contentDim)(SpatialConvolution(ngf*8, opt.contentDim, 4, 4)(enc5)))
  -- contentDim x 1 x 1

  local join = nn.JoinTable(2)({enc6, pose})
  
  -- contentDim+poseDim x 1 x 1
  local dec1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*8)(SpatialFullConvolution(opt.contentDim+opt.poseDim, ngf*8, 4, 4)(join)))
  -- ngf*8 x 4 x 4
  local dec2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*8)(SpatialFullConvolution(ngf*8*2, ngf*8, 4, 4, 2, 2, 1, 1)(nn.JoinTable(2)({dec1, enc5} ))))
  -- ngf*8 x 8 x 8
  local dec3 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*4)(SpatialFullConvolution(ngf*8*2, ngf*4, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec2, enc4} ))))
  -- ngf*4 x 16 x 16
  local dec4 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*2)(SpatialFullConvolution(ngf*4*2, ngf*2, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec3, enc3} ))))
  -- ngf*2 x 32 x 32
  local dec5 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf)(SpatialFullConvolution(ngf*2*2, ngf, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec4, enc2} ))))
  -- ngfx 64 x 64
  local dec6 = nn.Sigmoid()(SpatialFullConvolution(ngf*2, nc, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec5, enc1})))
  -- nc x 64 x 64
  local net = nn.gModule({x, pose}, {dec6, enc6})
  initModel(net)
  return net
end

local function vgg_layer(input, nin, nout)
  return nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(nout)(SpatialConvolution(nin, nout, 3, 3, 1, 1, 1, 1)(input)))
end

function makeUnetVGG()
  -- shared encoder
  local x = nn.Identity()()
  local pose = nn.Identity()()

  -- 128 -> 64
  local c1_1 = vgg_layer(nn.JoinTable(2)(x), nc*opt.nShare, 64)
  local c1_2 = vgg_layer(c1_1, 64, 64)
  local mp_1 = SpatialMaxPooling(2, 2, 2, 2)(c1_2)
  -- 64 -> 32
  local c2_1 = vgg_layer(mp_1, 64, 128)
  local c2_2 = vgg_layer(c2_1, 128, 128) --32x32
  local mp_2 = SpatialMaxPooling(2, 2, 2, 2)(c2_2)
  -- 32 -> 16
  local c3_1 = vgg_layer(mp_2, 128, 256)
  local c3_2 = vgg_layer(c3_1, 256, 256)
  local c3_3 = vgg_layer(c3_2, 256, 256) --16x16
  local mp_3 = SpatialMaxPooling(2, 2, 2, 2)(c3_3)
  -- 16 -> 8
  local c4_1 = vgg_layer(mp_3, 256, 512)
  local c4_2 = vgg_layer(c4_1, 512, 512)
  local c4_3 = vgg_layer(c4_2, 512, 512) --8x8x
  local mp_4 = SpatialMaxPooling(2, 2, 2, 2)(c4_3)
  -- 8 -> 4
  local c5_1 = vgg_layer(mp_4, 512, 512)
  local c5_2 = vgg_layer(c5_1, 512, 512)
  local c5_3 = vgg_layer(c5_2, 512, 512) -- 4x4
  local mp_5 = SpatialMaxPooling(2, 2, 2, 2)(c5_3)
  -- 4 -> 1
  local content = nn.Tanh()(SpatialBatchNormalization(opt.contentDim)(SpatialConvolution(512, opt.contentDim, 4, 4)(mp_5)))

  local join = nn.JoinTable(2)({content, pose})
  
  local comb = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(512)(SpatialFullConvolution(opt.contentDim+opt.poseDim, 512, 4, 4)(join)))

  -- 4 -> 8
  local up_1 = SpatialUpSamplingNearest(2)(comb) -- 4x4
  local d1_1 = vgg_layer(nn.JoinTable(2)({up_1, c5_3}), 512*2, 512)
  local d1_2 = vgg_layer(d1_1, 512, 512)
  local d1_3 = vgg_layer(d1_2, 512, 512)
  -- 8 -> 16
  local up_2 = SpatialUpSamplingNearest(2)(d1_3) --8x8
  local d2_1 = vgg_layer(nn.JoinTable(2)({up_2, c4_3}), 512*2, 512)
  local d2_2 = vgg_layer(d2_1, 512, 512)
  local d2_3 = vgg_layer(d2_2, 512, 256)
  -- 16 -> 32
  local up_3 = SpatialUpSamplingNearest(2)(d2_3) --16x16
  local d3_1 = vgg_layer(nn.JoinTable(2)({up_3, c3_3}), 256*2, 256)
  local d3_2 = vgg_layer(d3_1, 256, 256)
  local d3_3 = vgg_layer(d3_2, 256, 128)
  -- 32 -> 64
  local up_4 = SpatialUpSamplingNearest(2)(d3_3) --32x32
  local d4_1 = vgg_layer(nn.JoinTable(2)({up_4, c2_2}), 128*2, 128)
  local d4_2 = vgg_layer(d4_1, 128, 64)
  -- 64 -> 128
  local up_5 = SpatialUpSamplingNearest(2)(d4_2) --64x64
  local d5_1 = vgg_layer(nn.JoinTable(2)({up_5, c1_2}), 64*2, 64)
  local d5_2 = SpatialConvolution(64, nc, 3, 3, 1, 1, 1, 1)(d5_1)
  local out = nn.Sigmoid()(d5_2)

  -- nc x 64 x 64
  local net = nn.gModule({x, pose}, {out, content}) 
  initModel(net)

  return net
end

require 'torch'  
require 'nn'  
require 'nngraph'  
require 'cunn'  
require 'cudnn'  
require 'optim'  
require 'pl'  
require 'paths'  
require 'image'  
require 'utils'
succ, debugger = pcall(require,'fb.debugger')
 
----------------------------------------------------------------------  
-- parse command-line options  
opt = lapp[[  
  --learningRate     (default 0.002)             learning rate  
  --beta1            (default 0.5)               momentum term for adam
  -b,--batchSize     (default 100)               batch size  
  -g,--gpu           (default 0)                 gpu to use  
  --save             (default 'logs/')           base directory to save logs  
  --name             (default 'default')         checkpoint name
  --dataRoot         (default '/path/to/data/')  data root directory
  --optimizer        (default 'adam')            optimizer to train with
  --nEpochs          (default 300)               max training epochs  
  --seed             (default 1)                 random seed  
  --epochSize        (default 50000)             number of samples per epoch  
  --contentDim       (default 64)               dimensionality of noise space
  --poseDim          (default 16)               dimensionality of noise space
  --imageSize        (default 64)                size of image
  --dataset          (default moving_mnist)              dataset
  --movingDigits     (default 1)
  --cropSize         (default 227)               size of crop (for kitti only)
  --maxStep          (default 12)
  --nShare           (default 1)                 number of frame to use for content encoding
  --advWeight        (default 0)                 weight on adversarial scene discriminator loss 
  --normalize                                    if set normalize pose and action vectors to have unit norm
  --model            (default 'dcgan')
  --unet             (default 'dcgan')
  --nThreads         (default 0)                 number of dataloading threads
  --dataPool         (default 200)
  --dataWarmup       (default 10)  
]]  

opt.save = ('%s/%s/%s'):format(opt.save, opt.dataset, opt.name)
os.execute('mkdir -p ' .. opt.save .. '/gen/')
os.execute('mkdir -p ' .. opt.save .. '/swap/')

print(opt)
write_opt(opt)

assert(optim[opt.optimizer] ~= nil, 'unknown optimizer: ' .. opt.optimizer)
opt.optimizer = optim[opt.optimizer]

-- setup some stuff
torch.setnumthreads(1)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.gpu + 1)
print('<gpu> using device ' .. opt.gpu)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
math.randomseed(opt.seed)

local nc
if opt.dataset:find('suncg') or (opt.dataset == 'moving_mnist' and opt.movingDigits> 1) then
  nc = 3
else 
  nc = 1
end
opt.geometry = {nc, opt.imageSize, opt.imageSize}

if paths.filep(opt.save .. '/model.t7') then
  checkpoint = torch.load(opt.save .. '/model.t7') 
end
if checkpoint then
  netEP = checkpoint.netEP
  netEC = checkpoint.netEC
  netC = checkpoint.netC
  print('Loaded models from file')
else

  require(('models.%s_%d'):format(opt.model, opt.imageSize))
  require(('models.unet_%d'):format(opt.imageSize))
  if opt.unet == 'dcgan' then
    netEC = makeUnetDCGAN()
  elseif opt.unet == 'vgg' then
    netEC = makeUnetVGG()
  else
    assert(false)
  end
  netEP = makePoseEncoder()
  netC = makeSceneDiscriminator()

  print('Initialized models from scratch')
end

optimStateEP= {learningRate = opt.learningRate, beta=opt.beta1}
optimStateEC= {learningRate = opt.learningRate, beta=opt.beta1}
optimStateC = {learningRate = opt.learningRate, beta=opt.beta1}
netEC:cuda()
netEP:cuda()
netC:cuda()
params_EC, grads_EC= netEC:getParameters()
params_EP, grads_EP = netEP:getParameters()
params_C, grads_C = netC:getParameters()

rec_criterion = nn.MSECriterion()
rec_criterion:cuda()

sim_criterion = nn.MSECriterion()
sim_criterion:cuda()

bce_criterion = nn.BCECriterion()
bce_criterion:cuda()

local x = {}
for i=1,opt.maxStep do
  x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end
local x1, x2 = {}, {}
for i=1,opt.nShare do
  x1[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
  x2[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

local target = torch.CudaTensor(opt.batchSize)

function plot_pred(plot_x, fname)
  for i=1,opt.maxStep do
    x[i]:copy(plot_x[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  local offset = math.random(opt.nShare+1, opt.maxStep-opt.nShare)

  local hp = netEP:forward(x[offset])
  local pred, _ = unpack(netEC:forward({ x1, hp}))

  local N = math.min(20, opt.batchSize)

  local to_plot = {}
  for i=1,N do
    for ii=1,opt.nShare do
      table.insert(to_plot, x1[ii][i]:float())
    end
    table.insert(to_plot, x[offset][i]:float())
    table.insert(to_plot, pred[i]:float())
  end
  if opt.dataset:find('mnist') ~= nil then
    borderPlot(to_plot)
  end
  image.save(('%s/gen/%s_%d.png'):format(opt.save, fname, epoch), image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=(opt.nShare+2)*3})
end

function plot_swap(x_cpu, fname)
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  local offset = math.random(opt.nShare+1, opt.maxStep-opt.nShare)
  local N = math.min(opt.batchSize, 10)

  local hp_seq = {}
  for t=1,opt.maxStep do
    hp_seq[t] = netEP:forward(x[t]):clone()
    for i=2,N do
      hp_seq[t][i]:copy(hp_seq[t][1])
    end
  end

  local pred = {}
  for t=1,opt.maxStep do
    pred[t] = netEC:forward({x1, hp_seq[t]})[1]:clone()
  end

  local to_plot = {}
  for i=1,opt.nShare do
    table.insert(to_plot, torch.zeros(unpack(opt.geometry)))
  end
  for t=1,opt.maxStep do
    table.insert(to_plot, x[t][1]:float())
  end
  for i=1,N do
    for ii=1,opt.nShare do
      table.insert(to_plot, x1[ii][i]:float())
    end
    for j=1,opt.maxStep do
      table.insert(to_plot, pred[j][i]:float())
    end
  end
  
  if opt.dataset:find('mnist') then 
    borderPlot(to_plot)
  end
  image.save(('%s/swap/%s_%d.png'):format(opt.save, fname, epoch), image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=(opt.maxStep+opt.nShare)})
end

function test(x_cpu)
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  local offset = math.random(opt.nShare, opt.maxStep-opt.nShare)
  for i=1,opt.nShare do
    x2[i]:copy(x[i+offset])
  end

  local hp1 = netEP:forward(x[math.random(1, opt.maxStep)]):clone()
  local hp2 = netEP:forward(x[offset])
  local out1 = netEC:forward({x1, hp2})
  local pred, hc1 = out1[1]:clone(), out1[2]:clone()
  local hc2 = netEC:forward({x2, hp2})[2]

  -- ||h1 - h2||
  local latent_mse = sim_criterion:forward(hc1, hc2)

  -- ||D(hc1, hp2), x2||
  local pred_mse = rec_criterion:forward(pred, x[offset])

  -- scene discriminator loss
  target:fill(0.5)
  local out = netC:forward({hp1, hp2})
  local nll = bce_criterion:forward(out, target)
  local acc = out:gt(0.5):sum()

  return pred_mse, latent_mse, nll, acc
end

function train_scene_discriminator(x_cpu)
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end

  grads_C:zero()
  local bs = opt.batchSize
  local half = opt.batchSize/2
  target:sub(1,half):fill(1) --  1 if same scene
  target:sub(half+1,bs):fill(0) -- 0 if diff scene
  local offset1 = math.random(1, opt.maxStep)
  local offset2 = math.random(1, opt.maxStep)
  local hp1 = netEP:forward(x[offset1]):clone()
  local hp2 = netEP:forward(x[offset2]):clone()
  -- first half of batch pose vectors from same scene, second half randomly permute
  local rp = torch.linspace(1, opt.batchSize, opt.batchSize) 
  rp:sub(half+1,bs):copy(torch.randperm(half):add(half))
  local hp2 = hp2:index(1, rp:type('torch.LongTensor'))
  local out = netC:forward({hp1, hp2})
  local nll = bce_criterion:forward(out, target)
  local dout = bce_criterion:backward(out, target)
  netC:backward({hp1, hp2}, dout)
  local acc_same = out:sub(1,half):gt(0.5):sum()
  local acc_diff = out:sub(half+1,bs):lt(0.5):sum()

  opt.optimizer(function() return 0, grads_C end, params_C, optimStateC)

  return nll, acc_same, acc_diff
end

local dhc2 = torch.CudaTensor(opt.batchSize, opt.contentDim, 1, 1)
local dpred2 = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
  
function train(x_cpu)
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  local offset = math.random(opt.nShare+1, opt.maxStep-opt.nShare)
  for i=1,opt.nShare do
    x2[i]:copy(x[i+offset])
  end

  grads_EP:zero()
  grads_EC:zero()

  local hp1 = netEP:forward(x[math.random(1, opt.maxStep)]):clone()
  local hp2 = netEP:forward(x[offset])
  local hc2 = netEC:forward({x2, hp2})[2]:clone()
  local pred, hc1 = unpack(netEC:forward({x1, hp2}))

  -- minimize ||hc1 - hc2||
  local latent_mse = sim_criterion:forward(hc1, hc2)
  local dhc1_sim = sim_criterion:backward(hc1, hc2)

  -- maximize entropy of scene discrimintor output
  local dhp2_rec
  local nll = 0
  if opt.advWeight > 0 then
    target:fill(0.5)
    local out = netC:forward({hp1, hp2})
    bce_criterion:forward(out, target)
    local dout = bce_criterion:backward(out, target)
    dhp2_rec = netC:backward({hp1, hp2}, dout)[2]
  end

  -- minimize ||P(hc1, hp2), x2||
  local pred_mse = rec_criterion:forward(pred, x[offset])
  local dpred = rec_criterion:backward(pred, x[offset])
  local dhp2 = netEC:backward({x1, hp2}, {dpred, dhc1_sim})[2]
  if opt.advWeight > 0 then
    dhp2:add(opt.advWeight, dhp2_rec)
  end
  netEP:backward(x[offset], dhp2)

  opt.optimizer(function() return 0, grads_EC end, params_EC, optimStateEC)
  opt.optimizer(function() return 0, grads_EP end, params_EP, optimStateEP)

  return pred_mse, latent_mse
end

if opt.nThreads > 0 then
  dofile('data/threaded.lua')
else
  dofile(('data/%s.lua'):format(opt.dataset))
end
valLoader:plot()
cutorch.synchronize() 

plot_x_train = trainLoader:getBatch(opt.batchSize, opt.maxStep)
plot_x_val = valLoader:getBatch(opt.batchSize, opt.maxStep)

test_log = io.open(('%s/test.log'):format(opt.save), 'a')
train_log = io.open(('%s/train.log'):format(opt.save), 'a')

if checkpoint then
  best = checkpoint.best
  start_epoch = checkpoint.epoch+1
  total_iter = checkpoint.total_iter
  print('Starting training at epoch ' .. start_epoch)
else
  best = 1e10
  start_epoch = 0 
  total_iter = 0
end
epoch = start_epoch
while true do
  collectgarbage()
  collectgarbage()

  -- train
  print('\n<trainer> Epoch ' .. epoch )
  netEC:training()
  netEP:training()
  netC:training()
  local iter, pred_mse, latent_mse, sd_acc, sd_nll = 0, 0, 0, 0, 0
  local nTrain = opt.epochSize
  for i=1,nTrain,opt.batchSize do
    xlua.progress(i, nTrain)
    local batch = trainLoader:getBatch(opt.batchSize, opt.maxStep)
    local nll, acc_s, acc_d = train_scene_discriminator(batch)
    sd_nll = sd_nll + nll
    sd_acc = sd_acc + (acc_s+acc_d)
    local p_mse, l_mse = train(batch)
    pred_mse = pred_mse + p_mse
    latent_mse = latent_mse + l_mse
    iter=iter+1
    total_iter = total_iter + 1
  end
  print(('\n(%d)\tprediction mse = %.4f, latent mse = %.4f, scene disc acc = %.4f%%, scene disc nll = %.4f'):format(total_iter, pred_mse/iter, latent_mse/iter, 100*sd_acc/(opt.batchSize*iter), sd_nll/iter))

  train_log:write(('%.4f\t%.4f\t%.4f\t%.4f\n'):format(pred_mse/iter, latent_mse/iter, sd_nll/iter, 100*sd_acc/(opt.batchSize*iter)))
  train_log:flush()

  -- test
  netEC:evaluate()
  netEP:evaluate()
  netC:evaluate()
  local iter, pred_mse, latent_mse, sd_nll, sd_acc = 0, 0, 0, 0, 0
  local nTest = 1000
  for i=1,nTest,opt.batchSize do
    local p_mse, l_mse, nll, acc = test(valLoader:getBatch(opt.batchSize, opt.maxStep,  4))
    pred_mse = pred_mse + p_mse
    latent_mse = latent_mse + l_mse
    sd_nll = sd_nll + nll
    sd_acc = sd_acc + acc
    iter=iter+1
  end
  print(('\tprediction mse = %.4f, latent mse = %.4f'):format(pred_mse/iter, latent_mse/iter))

  test_log:write(('%.4f\t%.4f\t%.4f\t%.4f\n'):format(pred_mse/iter, latent_mse/iter, sd_nll/iter, 100*sd_acc/(opt.batchSize*iter)))
  test_log:flush()

  if pred_mse/iter < best then
    best = pred_mse / iter
    print(('Saving best model so far (pred mse = %.4f) %s/model_best.t7'):format(pred_mse/iter, opt.save))
    --torch.save(('%s/model_best.t7'):format(opt.save), {netEC=sanitize(netEC), netEP=sanitize(netEP), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end
 
  -- plot 
  --netEC:evaluate()
  --siameseEC:evaluate()
  plot_pred(plot_x_val, 'val')
  plot_pred(plot_x_train, 'train')

  plot_swap(valLoader:getBatch(opt.batchSize, opt.maxStep), 'val')
  plot_swap(trainLoader:getBatch(opt.batchSize, opt.maxStep), 'train')

  if epoch % 1 == 0 then
    print(('Saving model %s/model.t7'):format(opt.save))
    --torch.save(('%s/model.t7'):format(opt.save), {netC=sanitize(netC), netEC=sanitize(netEC), netEP=sanitize(netEP), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end
  epoch = epoch+1
  if epoch > opt.nEpochs then break end
end
train_log:close()
test_log:close()

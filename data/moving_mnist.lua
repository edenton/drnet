require 'torch'
require 'paths'
require 'image'

local MovingMNISTDataset = torch.class('MovingMNISTLoader')

if torch.getmetatable('dataLoader') == nil then
   torch.class('dataLoader')
end

local function getData(opt, datatype)
  local root_dir = paths.concat(opt.dataRoot, 'mnist/mnist.t7/')
  local data_path = paths.concat(root_dir, datatype .. '_32x32.t7')
  if not paths.filep(data_path) then
    -- get data
    local tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz' 
    os.execute(('mkdir %s/mnist'):format(opt.dataRoot))
    os.execute(('wget -P %s/mnist/ %s'):format(opt.dataRoot, tar))
    os.execute(('tar xvf %s/mnist/mnist.t7.tgz -C %s/mnist'):format(opt.dataRoot, opt.dataRoot))
  end
  local f = torch.load(data_path, 'ascii')
  local data = f.data:type(torch.getdefaulttensortype())
  local labels = f.labels:type("torch.IntTensor")
  return data, labels
end

function MovingMNISTDataset:__init(opt, data_type)
  local start, stop, data, labels
  if data_type == 'train' then
    start = 1
    stop = 55000
    data, labels = getData(opt, 'train')
  elseif data_type == 'val' then
    start = 55001
    stop = 60000
    data, labels = getData(opt, 'train')
  elseif data_type == 'test' then
    data, labels = getData(opt, 'test')
  end
  self.opt = opt or {}
  -- max 3 digits with current color scheme
  self.opt.movingDigits = math.min(self.opt.movingDigits or 1, 3)

  local nExample = data:size(1)
  local start = start or 1
  local stop = stop or nExample
  if stop > nExample then
    stop = nExample
  end 
  self.labels = labels[{{start, stop}}]
  self.data = data[{{start, stop}}]
  self.N = stop - start + 1 
  print('<mnist> loaded ' .. self.N .. ' examples.') 
end

function MovingMNISTDataset:size()
  return self.N 
end 

function MovingMNISTDataset:normalize()
  self.data:div(255)
end

local dirs = {4, -3, -2, -1, 1, 2, 3, 4}

function MovingMNISTDataset:getSequence(x)
  local t = x:size(1)
  x:zero()
  for ii = 1,self.opt.movingDigits do
    local idx = math.random(self.N)
    local sx = math.random(self.opt.imageSize-32)
    local sy = math.random(self.opt.imageSize-32)
    local dx = dirs[math.random(#dirs)]
    local dy = dirs[math.random(#dirs)]
    local k = 0
    for tt = 1,t do
      if sy < 1 then
        sy = 1
        dy = -dy --dirs[math.random((#dirs/2))+(#dirs/2)]
      elseif sy > self.opt.imageSize-32 then 
        sy = self.opt.imageSize - 32 
        dy = -dy --dirs[math.random((#dirs/2))]
      end

      if sx < 1 then
        sx = 1
        dx = -dx --dirs[math.random((#dirs/2))+(#dirs/2)]
      elseif sx > self.opt.imageSize -32 then 
        sx = self.opt.imageSize - 32 
        dx = -dx --dirs[math.random((#dirs/2))]
      end
      local ycor = {sy,sy+32-1}
      local xcor = {sx,sx+32-1}
      x[tt][{ ii, ycor, xcor }]:copy(self.data[idx])

      sy = sy + dy
      sx = sx + dx
    end
  end
  if self.opt.movingDigits > 1 then
    -- pick one color to be in front
    local c_order = torch.randperm(self.opt.movingDigits)
    for cc=2,self.opt.movingDigits do
      x[{ {}, c_order[cc]}][x[{ {}, c_order[1]}]:gt(0)] = 0
    end
  end
end

function MovingMNISTDataset:getBatch(n, T)
  local xx = torch.Tensor(T, unpack(self.opt.geometry))
  local x = {}
  for t=1,T do
    x[t] = torch.Tensor(n, unpack(self.opt.geometry))
  end
  for i = 1,n do
    self:getSequence(xx)
    for t=1,T do
      x[t][i]:copy(xx[t])
    end
  end 
  return x
end 

function MovingMNISTDataset:plotSeq(fname)
  print('plotting sequence: ' .. fname)
  local to_plot = {}
  local t = self.opt.T or 32 
  local n = 20
  local x = self:getBatch(n, t)
  for i = 1,n do
    for j = 1,t do
      table.insert(to_plot, x[j][i])
    end
  end 
  for i=1,#to_plot do
    to_plot[i][{ {}, {}, 1}]:fill(1)
    to_plot[i][{ {}, {}, 64}]:fill(1)
    to_plot[i][{ {}, 1, {}}]:fill(1)
    to_plot[i][{ {}, 64, {}}]:fill(1)
  end
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=t})
end

function MovingMNISTDataset:plot()
  local savedir = self.opt.save  .. '/data/'
  os.execute('mkdir -p ' .. savedir)
  self:plotSeq(savedir .. '/seq.png')
end

trainLoader = MovingMNISTLoader(opt, 'train')
trainLoader:normalize()
valLoader = MovingMNISTLoader(opt, 'val')
valLoader:normalize()

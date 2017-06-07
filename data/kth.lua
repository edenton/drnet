require 'torch'
require 'paths'
require 'image'
require 'utils'
debugger = require 'fb.debugger'


local KTHDataset = torch.class('KTHLoader')

if torch.getmetatable('dataLoader') == nil then
   torch.class('dataLoader')
end


function KTHDataset:__init(opt, data_type)
  self.data_type = data_type
  self.opt = opt or {}
  self.path = self.opt.dataRoot 
  self.data = torch.load(('%s/%s_meta.t7'):format(self.path, data_type))
  self.classes = {}
  for c, _ in pairs(self.data) do
    table.insert(self.classes, c)
  end

  print(('\n<loaded KTH %s data>'):format(data_type))
  local N = 0
  local shortest = 100
  local longest = 0
  for _, c in pairs(self.classes) do
    local n = 0
    local data = self.data[c]
    for i = 1,#data do 
      for d = 1,#data[i].indices do
        local len = data[i].indices[d][2] - data[i].indices[d][1] + 1
        if len < 0 then debugger.enter() end
        shortest = math.min(shortest, len)
        longest = math.max(longest, len)
      end
      n = n + self.data[c][i].n
      N = N + n
    end
    print(('%s: %d videos (%d total frames)'):format(c, #data, n)) 
  end
  self.N = N
  print('total frame: ' .. N)
  print(('min seq length = %d frames'):format(shortest))
  print(('max seq length = %d frames'):format(longest))
end

function KTHDataset:size()
  return self.N 
end 

function KTHDataset:getSequence(x, delta)
  local delta = math.random(1, delta or self.opt.delta or 1) 
  local c = self.classes[math.random(#self.classes)]
  local vid = self.data[c][math.random(#self.data[c])]
  local seq = math.random(#vid.indices)
  local seq_length = vid.indices[seq][2] - vid.indices[seq][1] + 1
  local basename = ('%s/%s/%s/'):format(self.path, c, vid.vid) 

  local T = x:size(1)
  while T*delta > seq_length do
    delta = delta-1
    if delta < 1 then return false end
  end

  local offset = math.random(seq_length-T*delta)
  local start = vid.indices[seq][1]
  for t = 1,T do
    local tt = start + offset+(t-1)*delta - 1
    local img = image.load(('%s/image-%03d_%dx%d.png'):format(basename, tt, self.opt.imageSize, self.opt.imageSize))[1]
    x[t]:copy(img)
  end
  return true, c_idx
end

function KTHDataset:getBatch(n, T, delta)
  local xx = torch.Tensor(T, unpack(self.opt.geometry))
  local x = {}
  for t=1,T do
    x[t] = torch.Tensor(n, unpack(self.opt.geometry))
  end
  for i = 1,n do
    while not self:getSequence(xx, delta) do
    end
    for t=1,T do
      x[t][i]:copy(xx[t])
    end
  end 
  return x
end 

function KTHDataset:plotSeq(fname)
  print('plotting sequence: ' .. fname)
  local to_plot = {}
  local t = 30 
  local n = 50
  for i = 1,n do
    local x = self:getBatch(1, t)
    for j = 1,t do
      table.insert(to_plot, x[j][1])
    end
  end 
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=t})
end

function KTHDataset:plot()
  local savedir = self.opt.save  .. '/data/'
  os.execute('mkdir -p ' .. savedir)
  self:plotSeq(savedir .. '/' .. self.data_type .. '_seq.png')
end

trainLoader = KTHLoader(opt_t or opt, 'train')
valLoader = KTHLoader(opt_t or opt, 'test')

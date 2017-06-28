-- derived from https://github.com/wojzaremba/lstm

local function lstm_layer(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(opt.rnnSize, 4*opt.rnnSize)(x)
  local h2h = nn.Linear(opt.rnnSize, 4*opt.rnnSize)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,opt.rnnSize)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function makeLSTM()
  local pose             = nn.Identity()()
  local content           = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i = {[0] = nn.Linear(opt.poseDim+opt.contentDim, opt.rnnSize)(nn.JoinTable(2)({pose, content}))}
  local next_s           = {}
  local split         = {prev_s:split(2 * opt.rnnLayers)}
  for layer_idx = 1, opt.rnnLayers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local next_c, next_h = lstm_layer(i[layer_idx - 1], prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end

  local gen_pose
  if opt.normalize then
    gen_pose = nn.Normalize(2)(nn.Tanh()(nn.Linear(opt.rnnSize, opt.poseDim)(i[opt.rnnLayers])))
  else
    gen_pose = nn.Tanh()(nn.Linear(opt.rnnSize, opt.poseDim)(i[opt.rnnLayers]))
  end

  base = nn.gModule({pose, content, prev_s}, {gen_pose, nn.Identity()(next_s)})
  initModel(base)
  base:cuda()
  params, grads = base:getParameters()

  lstm = {}
  lstm.params, lstm.grads = params, grads
  lstm.clones = clone_many(base, opt.T)
  lstm.base = base
  lstm.s = {}
  lstm.ds = {}
  for j = 0, opt.nPast+opt.nFuture do
    lstm.s[j] = {}
    for d = 1, 2 * opt.rnnLayers do
      lstm.s[j][d] = torch.CudaTensor(opt.batchSize, opt.rnnSize):fill(0)
    end
  end
  for d = 1, 2 * opt.rnnLayers do
    lstm.ds[d] = torch.CudaTensor(opt.batchSize, opt.rnnSize):fill(0) 
  end

  function lstm:reset_state()
    for d = 1, 2 * opt.rnnLayers do
      lstm.s[0][d]:zero()
    end
  end

  function lstm:reset_ds()
    for d = 1, #self.ds do
      self.ds[d]:zero()
    end
  end

  function lstm:fp_obs(pose, content)
    self:reset_state()
    local gen_pose = {}
    for i = 1, opt.nPast+opt.nFuture do
      local s = self.s[i - 1]
      gen_pose[i], self.s[i] = unpack(self.clones[i]:forward({pose[i], content, s}))
    end
    return gen_pose, pose
  end

  function lstm:fp_pred(pose, content)
    self:reset_state()
    local gen_pose = {}
    local in_pose = {}
    for i = 1, opt.nPast+opt.nFuture do
      local s = self.s[i - 1]
      if i <= opt.nPast then
        in_pose[i] = pose[i]:clone()
      else
        in_pose[i] = gen_pose[i-1]:clone()
      end
      gen_pose[i], self.s[i] = unpack(self.clones[i]:forward({in_pose[i], content, s}))
    end
    return gen_pose, in_pose
  end

  function lstm:bp(pose, content, dgen_pose)
    self:reset_ds()
    local ds = {}
    for i = opt.nPast+opt.nFuture, 1, -1 do
      local s = self.s[i - 1]
      local _, _, ds = unpack(self.clones[i]:backward({pose[i], content, s}, {dgen_pose[i], self.ds}))
      replace_table(self.ds, ds)
    end
  end
  return lstm
end

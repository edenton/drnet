function strsplit(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

function center_crop(x, crop)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = math.floor((x:size(2) - crop)/2)
  local sy = math.floor((x:size(3) - crop)/2)
  return image.crop(x, sy, sx, sy+crop, sx+crop)
end

function random_crop(x, crop, sx, sy)
  assert(x:dim() == 3)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = sx or math.random(0, x:size(2) - crop)
  local sy = sy or math.random(0, x:size(3) - crop)
  return image.crop(x, sy, sx, sy+crop, sx+crop), sx, sy
end

function adjust_meanstd(x, mean, std)
  for c = 1,3 do
    x[c]:add(-mean[c]):div(std[c])
  end
  return x
end

function normalize(x, min, max)
  local new_min = min or -1
  local new_max = max or 1
  local old_min, old_max = x:min(), x:max()
  local eps = 1e-7
  x:add(-old_min)
  x:mul(new_max - new_min)
  x:div(old_max - old_min + eps)
  x:add(new_min)
  return x
end

-- based on https://github.com/wojzaremba/lstm/blob/master/base.lua
function clone_many(net, T)
  local clones = {}
  local params, grads = net:parameters()
  local mem = torch.MemoryFile('w'):binary()
  mem:writeObject(net)
  for t = 1,T do
    local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGrads = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGrads[i]:set(grads[i])
    end 
    clones[t] = clone
    collectgarbage() 
  end 
  mem:close()
  return clones
end

function updateConfusion(confusion, output, targets)
  local correct = 0
  for i = 1,targets:nElement() do
    if targets[i] ~= -1 then
      local _, ind = output[i]:max(1)
      confusion:add(ind[1], targets[i])
      if ind[1] == targets[i] then
        correct = correct+1
      end
    end
  end
  return correct
end

function classResults(outputs, targets)
  local ind = {}
  local top1, top5, N = 0, 0, 0
  local _, sorted = outputs:float():sort(2, true)
  for i = 1,opt.batchSize do
    if targets[i] > 0 then -- has label
      ind[i] = 0
      N = N+1
      if sorted[i][1] == targets[i] then
        top1 = top1 + 1
        ind[i] = 1
      end
      for k = 1,5 do
        if sorted[i][k] == targets[i] then
          top5 = top5 + 1
          break
        end
      end
    end
  end
  return top1, top5, N, ind
end

function sanitize(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
      for name,field in pairs(val) do
         if torch.type(field) == 'cdata' then val[name] = nil end
         if name == 'homeGradBuffers' then val[name] = nil end
         if name == 'input_gpu' then val['input_gpu'] = {} end
         if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
         if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
         if (name == 'output' or name == 'gradInput') then
            if torch.isTensor(val[name]) then
               val[name] = field.new()
            end
         end
         if  name == 'buffer' or name == 'buffer2' or name == 'normalized'
         or name == 'centered' or name == 'addBuffer' then
            val[name] = nil
         end
      end
   end
   return net
end

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') or name:find('Linear') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end


function initModel(model)
  for _, m in pairs(model:listModules()) do
    weights_init(m)
  end
end

function isNan(x)
  return x:ne(x):sum() > 0 
end

function sampleNoise(z)
  if opt.noise == 'uniform' then
    z:uniform(-1, 1)
  else
    z:normal()
  end
end

function clone_table(t)
  local tt = {}
  for i=1,#t do
    tt[i] = t[i]:clone()
  end
end

function zero_table(t)
  for k, v in pairs(t) do
    t[k]:zero()
  end
end

function replace_table(t1, t2)
  for i=1,#t1 do
    t1[i]:copy(t2[i])
  end
end

function write_opt(opt)
  local opt_file = io.open(('%s/opt.log'):format(opt.save), 'w')
  for k, v in pairs(opt) do
    opt_file:write(('%s = %s\n'):format(k, v))
  end
  opt_file:close()
end


function borderPlot(to_plot, k)
  local k = k or 1
  local sx = to_plot[1]:size(2)
  local sy = to_plot[1]:size(3)
  for i=1,#to_plot do
    to_plot[i] = to_plot[i]:clone()
    to_plot[i][{ {}, {}, {1,k}}]:fill(1)
    to_plot[i][{ {}, {}, {sy-k+1,sy}}]:fill(1)
    to_plot[i][{ {}, {1,k}, {}}]:fill(1)
    to_plot[i][{ {}, {sx-k+1,sx}, {}}]:fill(1)
  end
end

function borderPlotRGB(to_plot, rgb)
  local nc = to_plot[1]:size(1)
  local sx = to_plot[1]:size(2)
  local sy = to_plot[1]:size(3)
  for i=1,#to_plot do
    local im 
    if nc == 1 then
      im = torch.expand(to_plot[i], 3, sx, sy):clone()
    else 
      im = to_plot[i]
    end
    to_plot[i] = im
    for c=1,3 do 
      to_plot[i][{ c, {}, 1}]:fill(rgb[c])
      to_plot[i][{ c, {}, sy}]:fill(rgb[c])
      to_plot[i][{ c, 1, {}}]:fill(rgb[c])
      to_plot[i][{ c, sx, {}}]:fill(rgb[c])
    end
  end
end

function borderPlotTensorRGB(x, rgb)
  local nc = x:size(1)
  local sx = x:size(2)
  local sy = x:size(3)
  local im 
  if nc == 1 then
    im = torch.expand(x, 3, sx, sy):clone()
  else 
    im = x
  end
  for c=1,3 do 
    im[{ c, {}, 1}]:fill(rgb[c])
    im[{ c, {}, sy}]:fill(rgb[c])
    im[{ c, 1, {}}]:fill(rgb[c])
    im[{ c, sx, {}}]:fill(rgb[c])
  end
  return im
end

function slice_table(input, start, end_)
  local result = {}

  local index = 1

  for i=start, end_ do
    result[index] = input[i]
    index = index + 1
  end

  return result
end


function extend_table(input, tail)
  for i=1, #tail do
    table.insert(input, tail[i])
  end
end

function find_index(t, e)
  for k, v in pairs(t) do
    if v == e then return k end
  end
end


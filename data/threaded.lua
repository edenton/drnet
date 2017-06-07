require 'data.threads'
opt.T = opt.T or opt.maxStep or (opt.Tpast+opt.Tfuture)
local opt_tt = opt -- need local var, opt is global
trainLoader = ThreadedDatasource(
    function()
        require 'pl'
        opt_t = tablex.copy(opt_tt)
        -- opt_t = opt_tt
        require(('data.%s'):format(opt_t.dataset))
        return trainLoader
    end, 
    {
      nThreads = opt_tt.nThreads,
      dataPool = math.ceil(opt_tt.dataPool / 10),
      dataWarmup = math.ceil(opt_tt.dataWarmup / 10),
    })
valLoader = ThreadedDatasource(
    function()
        require 'pl'
        opt_t = tablex.copy(opt_tt)
        require(('data.%s'):format(opt_t.dataset))
        return valLoader
    end, 
    {
      nThreads = opt_tt.nThreads,
      dataPool = math.ceil(opt_tt.dataPool / 10),
      dataWarmup = math.ceil(opt_tt.dataWarmup / 10),
    })

trainLoader:warm()
valLoader:warm()

cutorch.synchronize()

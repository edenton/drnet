--[[
   batchSize and T are fixed throughout.
--]]

local threads = require 'threads'

local ThreadedDatasource, parent = torch.class('ThreadedDatasource')

function ThreadedDatasource:__init(getDatasourceFun, params)
   self.nThreads = math.min(params.nThreads or 4, 2) -- XXX: fix
   local opt = opt

   self.pool_size = params.dataPool or opt.dataPool
   self.dataWarmup = params.dataWarmup or opt.dataWarmup
   self.pool = {}
   --threads.Threads.serialization('threads.sharedserialize') --TODO
   self.threads = threads.Threads(self.nThreads,
      function(threadid)
         require 'torch'
         require 'math'
         require 'os'
         opt_t = opt
         -- print(opt_t)
         torch.manualSeed(threadid*os.clock())
         math.randomseed(threadid*os.clock()*1.7)
         torch.setnumthreads(1)
         threadid_t = threadid
         datasource_t = getDatasourceFun()
      end)
   self.threads:synchronize()
   self.threads:specific(false)
end

function ThreadedDatasource:warm()
   print("Warming up batch pool...")
   for i = 1, self.dataWarmup do
      self:fetch_batch()

      -- don't let the job queue get too big
      if i % self.nThreads * 2 == 0 then
         self.threads:synchronize()
      end
      xlua.progress(i, self.dataWarmup)
   end

   -- get them working in the background
   for i = 1, self.nThreads * 2 do
      self:fetch_batch()
   end
end

function ThreadedDatasource:fetch_batch()
   self.threads:addjob(
      function()
         collectgarbage()
         collectgarbage()
          return table.pack(datasource_t:getBatch(opt_t.batchSize, opt_t.T))
      end,
      function(batch)
         collectgarbage()
         collectgarbage()
         if #self.pool < self.pool_size then
            table.insert(self.pool, batch)
         else
            local replacement_index = math.random(1, #self.pool)
            self.pool[replacement_index] = batch
         end
      end
   )
end

function ThreadedDatasource:plot()
   self.threads:addjob(
      function()
         collectgarbage()
         collectgarbage()
         datasource_t:plot()
      end,
      function()
      end
   )
  self.threads:dojob()
end

function ThreadedDatasource:getBatch()
   if self.threads:haserror() then
      print("ThreadedDatasource: There is an error in a thread")
      self.threads:terminate()
      os.exit(0)
   end

   -- queue has something for us
   -- dojob to put the newly loaded batch into the pool
   if self.threads.mainqueue.isempty == 0 then   
      self.threads:dojob()
      self:fetch_batch()
   end
   local batch_to_use = math.random(1, #self.pool)
   return unpack(self.pool[batch_to_use])
end

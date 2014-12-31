import os, numpy, redis, pickle, time
from cStringIO import StringIO

# Run experiment in separate process to handle crashes
from multiprocessing import Process, Queue

class Shared:
    def __init__(self, parent_folder, experiment):
        fdr = parent_folder if parent_folder  else '.'
        exp = experiment if experiment else 'experiment'

        self.folder = os.path.realpath(fdr + '/' + exp)
        self.waiting = exp + '-waiting'
        self.pending = exp + '-pending'
        self.running = exp + '-running'
        self.results = exp + '-results'

#         self.redis_server = 'deeplearn105.flickr.bf1.yahoo.com'
        # /raid/cfw
        # /raid/lfw

        self.redis_server = 'psglogin'
        # /data/shared/jculpepper

class Workr(Shared):
    def loop(self):
        r = redis.Redis(self.redis_server)
        print 'Ready'
        while True:
            # Wait for a job, and move it to running queue
            m = r.brpoplpush(self.pending, self.running, timeout=1)

            if m == None:
                # Trigger drivr to generate new experiment
                r.setex(self.waiting, '', 2)
            else:
                (job, params) = pickle.loads(m)
                print 'Running job %d from the grid: %s' % (job, params)
        
                start_time = time.time()
                value = self.run(job, **params)
                duration = time.time() - start_time
        
                print 'Result ', value
                r.lpush(self.results, pickle.dumps((job, value, duration)))
                r.lrem(self.running, m)

class GPUWorkr(Workr):
    def __init__(self, parent_folder=None, experiment=None, device=0):
        Workr.__init__(self, parent_folder, experiment)
        self.device = device

    def run(self, job, **params):
        return self.gpu_run(job, **params);

        q = Queue()
        p = Process(target=self.fork, args=(q, job), kwargs=params)
        p.start()
        p.join()
        if p.exitcode == 0:
            return q.get()
        else:
            return -1

    def fork(self, queue, job, **params):
        value = self.gpu_run(job, **params);
        queue.put(value)

class Drivr(Shared):
    def __init__(self, chooser, parent_folder=None, experiment=None, grid_size=20000, grid_seed=1):
        Shared.__init__(self, parent_folder, experiment)

        from ExperimentGrid import ExperimentGrid, EXPERIMENT_GRID_FILE
        from Locker import safe_delete
        from helpers import load_experiment


        self.chooser = chooser
        expt = load_experiment('config.pb')

        # Remove lock file, not used and can et stuck when python is killed
        safe_delete('%s.lock' % EXPERIMENT_GRID_FILE)
        print "hi"
        self.grid = ExperimentGrid(self.folder, expt.variable, grid_size, grid_seed)

    def loop(self):
        r = redis.Redis(self.redis_server)

        while True:
            print 'Waiting for workers'
            removed = 0
            while removed == 0:
                # Cannot use queue as redis doesn't have timeouts on items
                removed = r.delete(self.waiting)
                time.sleep(1)

            # Looking for new results
            while True:
                result = r.rpop(self.results)
                if result is None:
                    break
                (job_id, value, duration) = pickle.loads(result)
                if value != -1:
                    self.job_complete(job_id, value, duration)
                else:
                    self.job_broken(job_id)

            print 'Choosing next candidate...'
            job = self.next()
            params = self.params(job)
            print 'Selected job %d from the grid: %s' % (job, params)
            r.lpush(self.pending, pickle.dumps((job, params)))
            self.job_running(job)

            # Build table separately (multiple threads logging)
            s = StringIO()
            self.stats(s)
            print s.getvalue()
            # Update full file each time for consistency
            results = open(self.folder + '/results.txt', 'w')
            results.write(s.getvalue());
            results.close()

    def next(self):
        grid, values, durations = self.grid.get_grid()
        job_id = self.chooser.next(grid, values, durations, \
                              self.grid.get_candidates(), \
                              self.grid.get_pending(), \
                              self.grid.get_complete())

        # If the job_id is a tuple, then the chooser picked a new job.
        # We have to add this to our grid
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            job_id = self.grid.add_to_grid(candidate)

        return job_id

    def params(self, job_id):
        params = {}
        for param in self.grid.get_params(job_id):
            dbl_vals = param.dbl_val._values
            int_vals = param.int_val._values
            str_vals = param.str_val._values

            if len(dbl_vals) > 0:
                params[param.name] = numpy.array(dbl_vals)
            elif len(int_vals) > 0:
                params[param.name] = numpy.array(int_vals, dtype=int)
            elif len(str_vals) > 0:
                params[param.name] = str_vals

            if len(params[param.name]) == 1:
                params[param.name] = params[param.name][0]

        return params

    def job_running(self, job_id):
        self.grid.set_running(job_id)

    def job_complete(self, job_id, value, duration):
        self.grid.set_complete(job_id, value, duration)

    def job_broken(self, job_id):
        self.grid.set_broken(job_id)

    def stats(self, stream):
        rows = []

        for job in self.grid.get_complete():
            row = [job]
            row.append(self.grid.values[job])
            row.append(self.grid.durs[job])
            self._values(row, job)
            rows.append(row)
        rows.sort(key=lambda x: x[1])

        headers = ['Id', 'Result', 'Duration']
        for var in self.grid.vmap.variables:
            headers.append(var['name'])
        rows.insert(0, headers)

        for job in self.grid.get_pending():
            row = [job, 'Pending', 'Pending']
            self._values(row, job)
            rows.append(row)

        for job in self.grid.get_broken():
            row = [job, 'Failed', '']
            self._values(row, job)
            rows.append(row)

        return Drivr.table(stream, rows)
    
    def _values(self, row, job):
        params = self.params(job)
        for var in self.grid.vmap.variables:
            row.append(params[var['name']])

    @staticmethod
    def table(stream, rows):
        lens = []
        for i in range(len(rows[0])):
            col_lens = [len(str(x[i])) for x in rows]
            lens.append(max(col_lens))
        formats = []
        for i in range(len(rows[0])):
            if isinstance(rows[0][i], int):
                formats.append("%%%dd" % lens[i])
            else:
                formats.append("%%-%ds" % lens[i])
        pattern = " ".join(formats)
        for line in rows:
            stream.write(pattern % tuple(line) + '\n')


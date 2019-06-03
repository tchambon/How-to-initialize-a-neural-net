# This file contains utility function for this 
# repository notebooks
from fastai import *
from fastai.vision import *
import plotly.graph_objs as go
import plotly.graph_objs as go
import plotly.offline as pyoff


# HOOKS AND MONITORING SECTION
class Hook2():
    def __init__(self, m, f, *kwarg, **kwargs): 
        self.hook = m.register_forward_hook(partial(f, self))
        self.mod = m
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()
        
class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms:Collection[nn.Module], hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hooks = [Hook2(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self,i:int)->Hook2: return self.hooks[i]
    def __len__(self)->int: return len(self.hooks)
    def __iter__(self): return iter(self.hooks)
    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
        
        
def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])




def hook_stats(hook, mod, inp, outp):
    mean = outp.mean()
    std = outp.std()
    
  
    if mod.training:

        if not hasattr(hook,'stats'): hook.stats = ([],[])
        means,stds = hook.stats
            
        if mod.training:
            means.append(mean)
            stds .append(std)

def init_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats

    means.append(outp.data.mean().cpu())
    stds .append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(40,0,10)) #histc isn't implemented on the GPU

def append_stats_n(hist, hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    if mod.training:
        means.append(outp.data.mean().cpu())
        stds .append(outp.data.std().cpu())
        if hist:
            hists.append(outp.data.cpu().histc(40,0,10)) #histc isn't implemented on the GPU
            
def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[:2].sum(0)/h1.sum(0)    
    
def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

def stats_ratio_zeros(hooks):
    
    fig,axes = plt.subplots(len(hooks),1, figsize=(15,15))
    for ax,h in zip(axes.flatten(), hooks):
        ax.plot(get_min(h))
        ax.set_ylim(0,1)
    plt.tight_layout()
    
def get_prestat(learn, is_monitored_layer):
    output = []
    layers_monitored = find_modules(learn.model, is_monitored_layer)
    with Hooks(layers_monitored, init_stats) as first_stats: learn.pred_batch()
    for i, h in enumerate(first_stats):
        m,s, _ = h.stats
        output.append((m, s))   
        print(f'{str(h.mod)[:9]}: \n{m} / {s}\n')
    return output


# VISUALIZATION SECTION

def viz(hooks, zeros = False):
    fig,((ax0,ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(20,12))
    for h in hooks:
        ms,ss, hi = h.stats
        ax0.plot(ms[:10])
        ax1.plot(ss[:10])
        ax2.plot(ms)
        ax3.plot(ss)
    plt.legend(range(len(hooks)-1));

    if len(hi) > 0:
        fig,axes = plt.subplots(len(hooks),1, figsize=(20,12))
        for ax,h in zip(axes.flatten(), hooks):
            ax.imshow(get_hist(h), origin='lower')
            ax.axis('off')
        plt.tight_layout()
    
    
    if zeros:
        stats_ratio_zeros(hooks)





class RecordExp:
    def __init__(self):
        self.stock = []
        self.labels = []
        
    def add_exp(self, recorder, desc):
        self.stock.append(recorder)
        self.labels.append(desc)
        
        
    def plot(self):
        for exp in self.stock:
            exp.plot_losses()
            
    def plot_results_matplot(self):
        x_train = []
        x_val = []
        skip_start, skip_end = 0, 0
        
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,15))
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Batches processed')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Batches processed')
        #ax.legend(recorder.labels)
        
        for i, exp in enumerate(self.stock):
            losses = exp._split_list(exp.losses, skip_start, skip_end)
            iterations = exp._split_list(range_of(exp.losses), skip_start, skip_end)
            x_train.append((iterations, losses))
            
            ax1.plot(iterations, losses, label=self.labels[i])
            
            val_iter = exp._split_list_val(np.cumsum(exp.nb_batches), skip_start, skip_end)
            val_losses = exp._split_list_val(exp.val_losses, skip_start, skip_end)
            
            x_val.append((val_iter, val_losses))
            ax2.plot(val_iter, val_losses, label=self.labels[i])

        plt.legend()
        
        
   
    def plot_results_plotly(recorder, epochs, display_epochs=False):
#         x_train_iterations = []
#         x_train_losses = []

#         x_val_iterations = []
#         x_val_losses = []
        
        traces_train=[]
        traces_val=[]
        
        traces_met=[]
        min_train=math.inf
        max_train=0

        
        min_val=math.inf
        max_val=0
        
        
                
        min_met=math.inf
        max_met=0
        
        
        skip_start, skip_end = 0, 0
        
        
        #ax.legend(recorder.labels)
        
        for i, exp in enumerate(recorder.stock):
            losses = exp._split_list(exp.losses, skip_start, skip_end)
            iterations = exp._split_list(range_of(exp.losses), skip_start, skip_end)
            
            if min_train > min(losses):
                min_train = min(losses)
                
            if max_train < max(losses):
                max_train = max(losses)
            
            
#             x_train_iterations.append(iterations)
#             x_train_losses.append(losses)
            
            trace = go.Scatter(
                x = iterations,
                y = losses,
                name=recorder.labels[i]
            )
            
            traces_train.append(trace)
                        
            val_iter = exp._split_list_val(np.cumsum(exp.nb_batches), skip_start, skip_end)
            val_losses = exp._split_list_val(exp.val_losses, skip_start, skip_end)
            
            
            if min_val > min(val_losses):
                min_val = min(val_losses)
                
            if max_val < max(val_losses):
                max_val = max(val_losses)
            
#             x_val_iterations.append(val_iter)
#             x_val_losses.append(val_losses)
            
            trace = go.Scatter(
                x = val_iter,
                y = val_losses,
                name=recorder.labels[i]
            )
            
            traces_val.append(trace)
            
            metric_iter = val_iter
            metric_losses = exp._split_list_val([float(m[0]) for m in exp.metrics], skip_start, skip_end)
            
            
             
            if min_met > min(metric_losses):
                min_met = min(metric_losses)
                
            if max_met < max(metric_losses):
                max_met = max(metric_losses)
            
            trace = go.Scatter(
                x = metric_iter,
                y = metric_losses,
                name=recorder.labels[i]
            )
            
            traces_met.append(trace)
            

        
        data = traces_train
        
        it_per_epoch = len(recorder.stock[0].losses) / epochs
        
        epochs_lines = [i*it_per_epoch for i in range(1, epochs+1)]
        
        diff_train = (max_train - min_train)*0.05
        diff_val = (max_val - min_val)*0.05
        diff_met=(max_met - min_met)*0.05

        
        train_ann = [{'x':x_epochs, 'y': min_train-diff_train, 'text': f'E{i+1}', 'showarrow':False, 'arrowhead':7,
                       'ax': 0, 'ay': 40}
                      for i, x_epochs in enumerate(epochs_lines)]
        
        
        val_ann = [{'x':x_epochs, 'y': min_val-diff_val, 'text': f'E{i+1}', 'showarrow':False, 'arrowhead':7,
                       'ax': 0, 'ay': 40}
                      for i, x_epochs in enumerate(epochs_lines)]
        
        met_ann = [{'x':x_epochs, 'y': min_met-diff_met, 'text': f'E{i+1}', 'showarrow':False, 'arrowhead':7,
                       'ax': 0, 'ay': 40}
                      for i, x_epochs in enumerate(epochs_lines)]
        
        train_lines = [{'type': 'line', 'x0':x_epochs, 'x1':x_epochs, 'y0': min_train, 'y1':max_train, 'opacity': 0.7,
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'
            }}
                      for x_epochs in epochs_lines]
        
        val_lines = [{'type': 'line', 'x0':x_epochs, 'x1':x_epochs, 'y0': min_val, 'y1':max_val, 'opacity': 0.7,
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'}
                    }
                      for x_epochs in epochs_lines]
        
        
        met_lines = [{'type': 'line', 'x0':x_epochs, 'x1':x_epochs, 'y0': min_met, 'y1':max_met, 'opacity': 0.7,
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'}
                    }
                      for x_epochs in epochs_lines]
        
        
        
        layout_train = {
            'title': "Comparaison of train loss by init schemes",
            'annotations': train_ann,
            'shapes': train_lines,
            'xaxis': {'title': 'Iterations'},
            'yaxis': {'title': 'train loss'}
            
        }
        layout_val = {
            'title': "Comparaison of validation loss by init schemes",
            'annotations': val_ann,
            'shapes': val_lines,
            'xaxis': {'title': 'Iterations'},
            'yaxis': {'title': 'Val loss'}
            
        }
        
        layout_met = {
            'title': "Comparaison of metrics by init schemes",
            'annotations': met_ann,
            'shapes': met_lines,
            'xaxis': {'title': 'Iterations'},
            'yaxis': {'title': 'Accuracy'}
            
        }
        
        fig_train = {
            'data': traces_train,
            'layout': layout_train
        }
        
        if not display_epochs:
            fig_train.pop('layout', None)

        pyoff.iplot(fig_train, filename = 'results-train')
        
        
        fig_val = {
            'data': traces_val,
            'layout': layout_val
        }
        
        if not display_epochs:
            fig_val.pop('layout', None)   
            
        pyoff.iplot(fig_val, filename = 'results-val')
        
        
        fig_met = {
            'data': traces_met,
            'layout': layout_met
        }
        
        if not display_epochs:
            fig_met.pop('layout', None)   
            
        pyoff.iplot(fig_met, filename = 'results-met')
        
        
        

import sys, time
sys.path.insert(0, '/home/vlad/Cardio/generative/optimization')
sys.path.insert(0, '/home/vlad/Cardio/generative')
import torch
import numpy as np

def bench_all(impl_name, mod, device='cuda', dtype='fp32', epochs=256):
    torch.manual_seed(0)
    np.random.seed(0)
    n_samples = 1000
    theta = np.linspace(-4 * np.pi, 4 * np.pi, n_samples)
    z = np.linspace(-2, 2, n_samples)
    r = z**2 + 1
    x = r * np.sin(theta) + np.random.normal(0, 0.1, n_samples)
    y = r * np.cos(theta) + np.random.normal(0, 0.1, n_samples)
    dataset = torch.tensor(np.column_stack((x, y, z)), dtype=torch.float32)
    torch.ones(1, device=device); torch.cuda.synchronize() if device=='cuda' else None

    structure = mod.Structure(dataset, bayesian_network="all", bins=64, device=device, dtype=dtype)
    torch.optim.AdamW(mod.get_optim_groups(structure.model), lr=0.01, fused=True)  # warm one-time torch cost
    torch.cuda.synchronize() if device=='cuda' else None
    t0 = time.time()
    structure.fit(epochs=epochs, batch_size=256, lr=0.01, random_conditional_prob=0.4)
    torch.cuda.synchronize() if device=='cuda' else None
    fit_t = time.time() - t0

    grid2d = torch.stack(torch.meshgrid(torch.linspace(-5,5,64),torch.linspace(-4,4.5,50))).permute(1,2,0)
    grid2d_flat = grid2d.view(-1,2)
    xq = torch.linspace(dataset[:,0].min(),dataset[:,0].max(),128)

    with torch.no_grad():
        structure.conditional_dist([0.3,0.5],[0,1,2]); structure.generate(batch_size=1000)
        structure.partial_joint_log(grid2d_flat,[0,1]); structure.full_joint_log(dataset)
        torch.cuda.synchronize() if device=='cuda' else None
        def m(label, fn, reps=20):
            t0=time.time()
            for _ in range(reps): fn()
            torch.cuda.synchronize() if device=='cuda' else None
            print(f'    {label:30s} {(time.time()-t0)*1000/reps:8.3f} ms')
        cd = structure.conditional_dist([0.3,0.5],[0,1,2])
        m('conditional_dist', lambda: structure.conditional_dist([0.3,0.5],[0,1,2]))
        m('interp(128 pts).exp', lambda: cd(xq).detach().exp())
        m('generate(1000)', lambda: structure.generate(batch_size=1000))
        m('partial_joint_log', lambda: structure.partial_joint_log(grid2d_flat,[0,1]))
        m('full_joint_log', lambda: structure.full_joint_log(dataset))
    print(f'[{impl_name} {device} {dtype}] fit({epochs}ep): {fit_t:.3f}s = {fit_t*1000/epochs:.2f} ms/epoch')
    return fit_t

if __name__ == '__main__':
    import fast_bayesian_network_nn as fast
    import kemsekov_torch.bayesian_network_nn_v2 as orig
    print('=== CUDA fp32 ===')
    tf = bench_all('fast', fast, 'cuda', 'fp32')
    to = bench_all('orig', orig, 'cuda', 'fp32')
    print(f'fit speedup: {to/tf:.2f}x')
